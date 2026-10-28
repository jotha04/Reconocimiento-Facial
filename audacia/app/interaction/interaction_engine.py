"""
Motor Principal de Interacción y Decisiones de Saludo (InteractionEngine).
Implementa la arquitectura de dos capas:
1. Visual Session: Gestiona el grupo espacial continuo presente en ENGAGING.
2. GreetingBatch: Identifica lotes de participantes elegibles para recibir un saludo,
   permitiendo saludar a recién llegados sin repetir a quienes ya fueron saludados.
"""

import time
import logging
import threading
from typing import List, Optional, Dict, Any, Set

from audacia.app.tracking.track import Track, TrackState
from audacia.app.interaction.session import (
    InteractionSession, InteractionState, GroupType, Participant, GreetingBatch
)
from audacia.app.interaction.group_resolver import GroupResolver
from audacia.app.interaction.greeting_decision import GreetingDecisionEngine
from audacia.app.interaction.cooldown_manager import CooldownManager
from audacia.app.audio.audio_queue import AudioQueue
from audacia.app.attention.config import InteractionConfig, GreetingsConfig, AudioConfig

logger = logging.getLogger(__name__)


class InteractionEngine:
    """
    Gestiona la máquina de estados de interacción y emisión de lotes de saludos.
    """

    def __init__(
        self,
        interaction_config: InteractionConfig,
        greetings_config: GreetingsConfig,
        audio_config: AudioConfig,
        audio_queue: Optional[AudioQueue] = None,
        attention_config: Optional[Any] = None
    ):
        self.config = interaction_config
        self.greetings_config = greetings_config
        self.audio_config = audio_config
        self.attention_config = attention_config

        self.group_resolver = GroupResolver(self.config, attention_config=self.attention_config)
        self.decision_engine = GreetingDecisionEngine(self.greetings_config)
        self.cooldown_manager = CooldownManager(
            person_cooldown_seconds=self.config.person_cooldown_seconds,
            unknown_cooldown_seconds=self.config.unknown_cooldown_seconds
        )
        self.audio_queue = audio_queue or AudioQueue(config=self.audio_config)

        self.current_session: Optional[InteractionSession] = None
        self._next_session_id = 1
        self._next_batch_id = 1

        self.last_blocked_reason: str = ""
        self.last_batch_reason: str = "NONE"
        self.total_greetings_emitted = 0

        # BUG-002 fix: retry tracking for enqueue failures
        self._enqueue_retry_count = 0
        self._enqueue_retry_timeout_s = 15.0  # Time-based timeout for enqueue retry
        self._enqueue_first_attempt_at: Optional[float] = None
        self._lock = threading.Lock()

        # BUG-003 fix: timeout for greeting states
        self._greeting_state_entered_at: Optional[float] = None
        self._greeting_state_timeout_s = 15.0

        # Diagnóstico y telemetría de grupo
        self._current_tracks: List[Track] = []

    def update(self, tracks: List[Track]) -> Optional[InteractionSession]:
        """
        Ciclo principal de actualización del InteractionEngine por frame.
        """
        now = time.monotonic()
        post_lock_actions = []
        
        # Concurrency safety: main thread vs audio worker callbacks
        self._lock.acquire()
        try:
            self._current_tracks = tracks
            # 1. Extraer participantes potenciales (tracks exclusivamente en estado ENGAGING)
            engaging_candidates = self.group_resolver.extract_engaging_candidates(tracks)

            # 2. Si no hay sesión activa y aparecen participantes en ENGAGING -> Iniciar nueva sesión
            if self.current_session is None:
                if engaging_candidates:
                    self.current_session = InteractionSession(
                        session_id=self._next_session_id,
                        created_at=now,
                        last_active_at=now,
                        state=InteractionState.CANDIDATE_GROUP,
                        stabilization_start_time=now,
                        resolution_start_time=now
                    )
                    self._next_session_id += 1
                    logger.info(f"[DIAGNOSTICS] Nueva sesión visual iniciada: #{self.current_session.session_id} en {now:.3f}")
                    self._log_group_state(self.current_session, {c.track_id for c in engaging_candidates})
                else:
                    return None
    
            session = self.current_session
    
            # BUG-001 fix: compute active_track_ids BEFORE the loop (not during)
            active_track_ids: Set[int] = {c.track_id for c in engaging_candidates}
    
            # Sincronizar participantes activos en la sesión visual
            for cand in engaging_candidates:
                if cand.track_id not in session.participants:
                    # BUG-001 fix: Anti-flicker reconnection with time limit, identity check,
                    # and single consumption
                    self._try_anti_flicker_reconnect(session, cand, active_track_ids)
                    session.participants[cand.track_id] = cand
                else:
                    # Actualizar información dinámica (por si cambió de NOT_EVALUATED a KNOWN)
                    p = session.participants[cand.track_id]
                    was_unknown = not p.is_known
                    p.identity_label = cand.identity_label
                    p.identity_decision = cand.identity_decision
                    p.person_id = cand.person_id or p.person_id
                    p.visitor_id = cand.visitor_id or p.visitor_id
                    p.recognition_similarity = cand.recognition_similarity
                    p.center_x = cand.center_x
                    p.center_y = cand.center_y
                    p.bbox_width = cand.bbox_width
                    p.bbox_height = cand.bbox_height
                    if cand.embedding is not None:
                        p.embedding = cand.embedding
                    p.last_seen_at = time.monotonic()
                    if was_unknown and p.is_known:
                        if not self.cooldown_manager.is_person_in_cooldown(p.person_id, p.identity_label):
                            p.greeted_in_session = False
    
            # 3. Mantener temporizador de actividad si hay participantes presentes
            if engaging_candidates:
                session.last_active_at = now
    
            # 4. Manejo de fin de sesión por inactividad
            if not engaging_candidates:
                if session.idle_seconds > self.config.session_timeout_seconds:
                    logger.info(
                        f"Sesión #{session.session_id} cerrada tras {session.idle_seconds:.1f}s de inactividad."
                    )
                    session.state = InteractionState.NO_INTERACTION
                    self.current_session = None
                    return session
    
            # 5. Lógica de saludo: Grupo Inicial vs Nuevos Participantes (GreetingBatch)
            state = session.state
    
            # A. FASE INICIAL DE LA SESIÓN: Estabilizar el primer grupo
            if state in (InteractionState.CANDIDATE_GROUP, InteractionState.GROUP_STABILIZING, InteractionState.IDENTITIES_RESOLVING):
                is_ready, is_timeout, new_stab, new_res, reason = self.group_resolver.check_stabilization(
                    current_candidates=engaging_candidates,
                    tracked_participants=session.participants,
                    stabilization_start_time=session.stabilization_start_time,
                    resolution_start_time=session.resolution_start_time,
                    all_tracks=tracks
                )
                session.stabilization_start_time = new_stab
                session.resolution_start_time = new_res
                self.last_blocked_reason = reason
    
                if not self.group_resolver.are_identities_resolved(engaging_candidates):
                    session.state = InteractionState.IDENTITIES_RESOLVING
                else:
                    session.state = InteractionState.GROUP_STABILIZING
    
                if is_ready:
                    session.state = InteractionState.GROUP_CONFIRMED
                    session.stabilization_start_time = None
                    session.resolution_start_time = None
    
                    present_participants = [
                        p for p in session.participants.values()
                        if p.track_id in active_track_ids
                    ]
                    act = self._dispatch_greeting_batch(session, present_participants, reason="INITIAL_GROUP")
                    if act: post_lock_actions.append(act)
    
            # B. FASE ACTIVA DE LA SESIÓN: Monitorear participantes recién llegados (GreetingBatch)
            elif state == InteractionState.ACTIVE_ENGAGEMENT:
                # BUG-002 fix: Gestión prioritaria de pending_batch
                if session.pending_batch:
                    p_batch = session.pending_batch
                    batch_track_ids = {p.track_id for p in p_batch.participants}
                    current_ungreeted_track_ids = {
                        p.track_id for p in session.participants.values()
                        if p.track_id in active_track_ids and not p.greeted_in_session
                    }
                    
                    if batch_track_ids != current_ungreeted_track_ids:
                        # El grupo cambió (alguien se fue o alguien nuevo entró). Cancelar el lote pendiente obsoleto.
                        logger.info("El grupo cambió mientras se esperaba el audio. Cancelando lote pendiente obsoleto.")
                        p_batch.is_completed = True
                        p_batch.completed_at = now
                        session.batch_history.append(p_batch)
                        session.pending_batch = None
                        self._enqueue_retry_count = 0
                        self._enqueue_first_attempt_at = None
                    elif self._enqueue_first_attempt_at and (now - self._enqueue_first_attempt_at) > self._enqueue_retry_timeout_s:
                        logger.error(
                            f"Lote pendiente #{p_batch.batch_id} descartado tras timeout de {self._enqueue_retry_timeout_s}s. "
                            "Participantes NO marcados como saludados."
                        )
                        p_batch.is_completed = True
                        p_batch.completed_at = now
                        session.batch_history.append(p_batch)
                        session.pending_batch = None
                        self._enqueue_retry_count = 0
                        self._enqueue_first_attempt_at = None
                    else:
                        if not self.audio_queue.is_playing():
                            act = self._try_enqueue_pending_batch(session)
                            if act: post_lock_actions.append(act)
                    return session
    
                if self.audio_queue.is_playing():
                    # Audio ocupado: esperar a que termine antes de evaluar nuevos despachos
                    return session
    
                ungreeted = [
                    p for p in session.participants.values()
                    if p.track_id in active_track_ids and not p.greeted_in_session
                ]
    
                if ungreeted:
                    # BUG-007 partial fix: separate resolved from pending, don't let pending block resolved
                    resolved = [p for p in ungreeted if not p.is_pending]
                    pending = [p for p in ungreeted if p.is_pending]
    
                    if resolved:
                        # Breve ventana de estabilización para incorporar si llegan dos juntos
                        if session.stabilization_start_time is None:
                            session.stabilization_start_time = now
                            logger.info(f"[DIAGNOSTICS] stabilization_start_time INICIADO para NEW_PARTICIPANTS en {now:.3f}")
                            self._log_group_state(session, active_track_ids)
    
                        elapsed = now - session.stabilization_start_time
                        if elapsed >= 0.50:  # 500 ms de gracia para agrupar múltiples recién llegados
                            session.stabilization_start_time = None
                            act = self._dispatch_greeting_batch(session, resolved, reason="NEW_PARTICIPANTS")
                            if act: post_lock_actions.append(act)
                        else:
                            rem = 0.50 - elapsed
                            self.last_blocked_reason = f"Estabilizando nuevos participantes ({rem:.2f}s)"
                    elif pending:
                        self.last_blocked_reason = "Identidad de nuevo participante en resolución..."
    
                # BUG-013 fix: reset stabilization timer when no ungreeted participants
                if not ungreeted and session.stabilization_start_time is not None:
                    session.stabilization_start_time = None
    
            # C. Transiciones de estado de audio
            elif state == InteractionState.GREETING_QUEUED:
                # BUG-003 fix: track when we entered a greeting state
                if self._greeting_state_entered_at is None:
                    self._greeting_state_entered_at = now
    
                if self.audio_queue.is_playing():
                    session.state = InteractionState.GREETING_PLAYING
                elif now - self._greeting_state_entered_at > self._greeting_state_timeout_s:
                    logger.error(
                        f"Timeout de {self._greeting_state_timeout_s}s en GREETING_QUEUED. "
                        "Forzando transición a ACTIVE_ENGAGEMENT."
                    )
                    self._force_recover_greeting_state(session)
    
            elif state == InteractionState.GREETING_PLAYING:
                # BUG-003 fix: track when we entered a greeting state
                if self._greeting_state_entered_at is None:
                    self._greeting_state_entered_at = now
    
                if not self.audio_queue.is_playing() and (session.active_batch is None or session.active_batch.is_completed):
                    session.state = InteractionState.ACTIVE_ENGAGEMENT
                    self._greeting_state_entered_at = None
                elif now - self._greeting_state_entered_at > self._greeting_state_timeout_s:
                    logger.error(
                        f"Timeout de {self._greeting_state_timeout_s}s en GREETING_PLAYING. "
                        "Forzando transición a ACTIVE_ENGAGEMENT."
                    )
                    self._force_recover_greeting_state(session)

            return session
        finally:
            self._lock.release()
            for act in post_lock_actions:
                act()

    def _log_group_state(
        self,
        session: InteractionSession,
        active_track_ids: Optional[Set[int]] = None,
        candidate_participants: Optional[List[Participant]] = None,
        eligible: Optional[List[Participant]] = None
    ):
        """
        Registra telemetría de diagnóstico para la ventana y estado del grupo.
        """
        now = time.monotonic()
        if candidate_participants is not None:
            active_participants = candidate_participants
        else:
            if active_track_ids is None:
                active_track_ids = {t.track_id for t in self._current_tracks if t.state == TrackState.ENGAGING}
            active_participants = [
                p for p in session.participants.values()
                if p.track_id in active_track_ids
            ]

        if eligible is None:
            eligible, _ = self.cooldown_manager.filter_eligible_for_greeting(active_participants)

        pending_ids = [
            t.track_id for t in self._current_tracks
            if t.identity_decision == "NOT_EVALUATED" and t.state in (TrackState.ENGAGING, TrackState.ATTENTION_CANDIDATE)
        ]
        stab_start = f"{session.stabilization_start_time:.3f}" if session.stabilization_start_time is not None else "None"
        stab_elapsed = f"{now - session.stabilization_start_time:.3f}s" if session.stabilization_start_time is not None else "None"
        group_snapshot = [
            {"track_id": p.track_id, "label": p.identity_label, "decision": p.identity_decision}
            for p in active_participants
        ]

        logger.info(
            f"[GROUP]\n"
            f"  active participants={[p.track_id for p in active_participants]}\n"
            f"  eligible participants={[p.track_id for p in eligible]}\n"
            f"  pending recognition={pending_ids}\n"
            f"  stabilization_start={stab_start}\n"
            f"  stabilization_elapsed={stab_elapsed}\n"
            f"  group snapshot={group_snapshot}"
        )

    def _try_anti_flicker_reconnect(
        self,
        session: InteractionSession,
        cand: Participant,
        active_track_ids: Set[int]
    ):
        """
        BUG-001 fix: Reconexión anti-flicker con límite temporal (≤1.5s),
        verificación de identidad compatible, y consumo único.
        """
        now = time.monotonic()
        max_absence_seconds = 1.5

        for prev_p in session.participants.values():
            # Solo considerar participantes ya saludados que no estén actualmente presentes
            if not prev_p.greeted_in_session:
                continue
            if prev_p.track_id in active_track_ids:
                continue
            # Ya fue consumido por una reconexión anterior
            if prev_p.reconnected_to is not None:
                continue

            # Límite temporal: solo si desapareció hace poco
            absence_duration = now - prev_p.last_seen_at
            if absence_duration > max_absence_seconds:
                continue

            # Proximidad espacial
            if abs(cand.center_x - prev_p.center_x) >= 0.20:
                continue

            # Verificación de identidad: si ambos están resueltos, deben ser compatibles
            if not cand.is_pending and not prev_p.is_pending:
                # KNOWN vs KNOWN: deben tener la misma identidad
                if cand.is_known and prev_p.is_known:
                    if cand.identity_label.lower() != prev_p.identity_label.lower():
                        continue
                # KNOWN vs UNKNOWN o viceversa: no reconectar
                elif cand.is_known != prev_p.is_known:
                    continue
                # UNKNOWN vs UNKNOWN: si ambos tienen visitor_id asignado, deben coincidir
                elif cand.visitor_id is not None and prev_p.visitor_id is not None:
                    if cand.visitor_id != prev_p.visitor_id:
                        continue

            # Si el candidato aún es NOT_EVALUATED, no reconectar — esperar
            if cand.is_pending:
                continue

            # Reconexión válida: transferir estado y consumir
            cand.greeted_in_session = True
            cand.greeted_at = prev_p.greeted_at
            prev_p.reconnected_to = cand.track_id  # consumir: solo una reconexión
            logger.info(
                f"Track #{cand.track_id} asociado como reconexión de track #{prev_p.track_id} "
                f"en sesión #{session.session_id} (ausencia: {absence_duration:.2f}s, saludo preservado)."
            )
            return

    def _force_recover_greeting_state(self, session: InteractionSession):
        """
        BUG-003 fix: Force recovery from stuck GREETING_QUEUED/PLAYING states.
        Clears the active and pending batches without marking participants as greeted.
        """
        batch = session.active_batch or session.pending_batch
        if batch and not batch.is_completed:
            batch.is_completed = True
            batch.was_force_recovered = True
            batch.completed_at = time.monotonic()
            if batch not in session.batch_history:
                session.batch_history.append(batch)
            logger.warning(
                f"Lote #{batch.batch_id} recuperado por timeout de seguridad. "
                "Participantes NO marcados como saludados (se reintentará)."
            )
        session.active_batch = None
        session.pending_batch = None
        session.state = InteractionState.ACTIVE_ENGAGEMENT
        self._greeting_state_entered_at = None
        self._enqueue_retry_count = 0
        self._enqueue_first_attempt_at = None

    def _try_enqueue_pending_batch(self, session: InteractionSession):
        """
        BUG-002 fix: Reintenta encolar un lote pendiente de saludo cuando la cola se libera.
        """
        batch = session.pending_batch
        if not batch or batch.is_completed:
            return

        now = time.monotonic()
        session_id = session.session_id

        def on_started(txt):
            batch.started_at = time.monotonic()
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session.state = InteractionState.GREETING_PLAYING

        def on_completed(txt, success=True):
            with self._lock:
                # BUG-003 fix: Idempotency guard against race with force recovery
                if batch.is_completed:
                    logger.warning(
                        f"Callback on_completed recibido para lote #{batch.batch_id} "
                        "que ya estaba completado o recuperado por timeout. Ignorando por idempotencia."
                    )
                    return

                if success:
                    session.mark_batch_completed(batch)
                    self.cooldown_manager.record_greeting(batch.participants)
                    self.total_greetings_emitted += 1
                else:
                    logger.warning(f"Audio fallido para lote #{batch.batch_id}. Participantes NO marcados.")
                    batch.is_completed = True
                    batch.completed_at = time.monotonic()
                    if batch not in session.batch_history:
                        session.batch_history.append(batch)
                    if session.active_batch is batch:
                        session.active_batch = None
                    if session.pending_batch is batch:
                        session.pending_batch = None

                if self.current_session and self.current_session.session_id == session_id:
                    self.current_session.state = InteractionState.ACTIVE_ENGAGEMENT
                    self._greeting_state_entered_at = None

        def perform_enqueue():
            dispatch_time = time.monotonic()
            logger.info(f"[DIAGNOSTICS] Llamando enqueue_greeting() en {dispatch_time:.3f} para lote #{batch.batch_id}")
            enqueued = self.audio_queue.enqueue_greeting(
                batch.greeting_text,
                on_started=on_started,
                on_completed=on_completed
            )
            with self._lock:
                if enqueued:
                    session.active_batch = batch
                    session.pending_batch = None
                    session.state = InteractionState.GREETING_QUEUED
                    self._greeting_state_entered_at = time.monotonic()
                    self._enqueue_retry_count = 0
                    self._enqueue_first_attempt_at = None
                    logger.info(f"Lote pendiente #{batch.batch_id} encolado con éxito tras reintento.")
                else:
                    self._enqueue_retry_count += 1
                    logger.debug(
                        f"Audio aún ocupado al reintentar lote #{batch.batch_id} "
                        f"(intento {self._enqueue_retry_count})."
                    )
        return perform_enqueue

    def _dispatch_greeting_batch(
        self,
        session: InteractionSession,
        candidate_participants: List[Participant],
        reason: str
    ):
        """
        Evalúa cooldowns y emite un GreetingBatch exclusivamente para los participantes elegibles.
        """
        now = time.monotonic()
        eligible, blocked = self.cooldown_manager.filter_eligible_for_greeting(candidate_participants)

        # BUG-004 partial fix: Para los participantes bloqueados por cooldown de IDENTIDAD (KNOWN),
        # marcarlos como saludados. Para bloqueados por huella espacial (UNKNOWN), NO marcar —
        # solo esperar a que expire la huella.
        for b in blocked:
            if b.is_known:
                b.greeted_in_session = True
                b.greeted_at = now
                logger.info(f"Participante '{b.identity_label}' en cooldown de identidad. Saludo omitido.")
            elif getattr(b, 'cooldown_reason', None) in ("TRACK_COOLDOWN", "APPEARANCE_COOLDOWN"):
                # Si el mismo track_id o misma apariencia facial ya fue saludada, no repetir saludo en esta sesión
                b.greeted_in_session = True
                b.greeted_at = now
                logger.info(f"Participante UNKNOWN #{b.track_id} en {b.cooldown_reason}. Saludo omitido en esta sesion.")
            else:
                logger.info(f"Participante UNKNOWN #{b.track_id} en cooldown por huella. Esperando expiración.")

        # Telemetría de diagnóstico requerida: [GROUP]
        self._log_group_state(session, candidate_participants=candidate_participants, eligible=eligible)

        if not eligible:
            physical_tracks = [t for t in self._current_tracks if t.state != TrackState.DISENGAGED]
            logger.info(
                f"[GREETING DECISION]\n"
                f"  * physical_group_size={len(physical_tracks)}\n"
                f"  * eligible_size=0\n"
                f"  * greeting_mode=BLOCKED_BY_COOLDOWN\n"
                f"  * participants_in_text=None"
            )
            # Check if there are unknown participants waiting for footprint cooldown
            unknown_waiting = any(not b.is_known and not b.greeted_in_session for b in blocked)
            if unknown_waiting:
                self.last_blocked_reason = "Participante(s) UNKNOWN esperando expiración de huella"
            else:
                self.last_blocked_reason = "Participante(s) bloqueado(s) por cooldown activo"
            session.state = InteractionState.ACTIVE_ENGAGEMENT
            return

        # Re-sincronizar participantes con el estado más reciente de la sesión (por si hubo rescate oportuno de UNKNOWN a KNOWN)
        for p in candidate_participants:
            if p.track_id in session.participants:
                sp = session.participants[p.track_id]
                p.identity_label = sp.identity_label
                p.identity_decision = sp.identity_decision
                p.person_id = sp.person_id or p.person_id
                p.visitor_id = getattr(sp, "visitor_id", None) or getattr(p, "visitor_id", None)
                p.recognition_similarity = sp.recognition_similarity

        # Generar texto de saludo basándose en todo el grupo presente para conservar la pluralidad correcta
        group_type, text = self.decision_engine.generate_greeting(candidate_participants)
        physical_tracks = [t for t in self._current_tracks if t.state != TrackState.DISENGAGED]
        logger.info(
            f"[GREETING DECISION]\n"
            f"  * physical_group_size={len(physical_tracks)}\n"
            f"  * eligible_size={len(eligible)}\n"
            f"  * greeting_mode={group_type.name}\n"
            f"  * participants_in_text='{text}'"
        )
        batch = GreetingBatch(
            batch_id=self._next_batch_id,
            participants=eligible,
            group_type=group_type,
            greeting_text=text,
            reason=reason,
            created_at=now
        )
        self._next_batch_id += 1
        session.active_batch = batch
        self.last_batch_reason = reason

        logger.info(
            f"[DIAGNOSTICS] Lote de saludo #{batch.batch_id} CREADO en {now:.3f} ({reason}): "
            f"Participantes={[p.identity_label for p in eligible]}, Texto='{text}'"
        )

        session.state = InteractionState.GREETING_QUEUED
        self._greeting_state_entered_at = now
        batch.queued_at = now

        session_id = session.session_id

        def on_started(txt):
            st = time.monotonic()
            logger.info(f"[DIAGNOSTICS] Audio on_started recibido en {st:.3f} para lote #{batch.batch_id}")
            batch.started_at = st
            # BUG-011 fix: verify session_id before modifying state
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session.state = InteractionState.GREETING_PLAYING

        def on_completed(txt, success=True):
            with self._lock:
                # BUG-003 fix: Idempotency guard against race with force recovery
                if batch.is_completed:
                    logger.warning(
                        f"Callback on_completed recibido para lote #{batch.batch_id} "
                        "que ya estaba completado o recuperado por timeout. Ignorando por idempotencia."
                    )
                    return

                if success:
                    session.mark_batch_completed(batch)
                    self.cooldown_manager.record_greeting(batch.participants)
                    self.total_greetings_emitted += 1
                else:
                    # BUG-016 fix: failed audio should NOT mark participants as greeted
                    logger.warning(f"Audio fallido para lote #{batch.batch_id}. Participantes NO marcados.")
                    batch.is_completed = True
                    batch.completed_at = time.monotonic()
                    if batch not in session.batch_history:
                        session.batch_history.append(batch)
                    if session.active_batch is batch:
                        session.active_batch = None
                    if session.pending_batch is batch:
                        session.pending_batch = None
                if self.current_session and self.current_session.session_id == session_id:
                    self.current_session.state = InteractionState.ACTIVE_ENGAGEMENT
                    self._greeting_state_entered_at = None

        def perform_enqueue():
            dispatch_time = time.monotonic()
            logger.info(f"[DIAGNOSTICS] Llamando enqueue_greeting() en {dispatch_time:.3f} para lote #{batch.batch_id}")
            enqueued = self.audio_queue.enqueue_greeting(text, on_started=on_started, on_completed=on_completed)
            with self._lock:
                if not enqueued:
                    # Conservar como pending_batch para reintentar en los siguientes frames
                    self._enqueue_retry_count = 1
                    self._enqueue_first_attempt_at = time.monotonic()
                    session.pending_batch = batch
                    session.active_batch = None
                    self._greeting_state_entered_at = None
                    session.state = InteractionState.ACTIVE_ENGAGEMENT
                    logger.debug(
                        f"Audio ocupado en primer despacho. Guardando lote #{batch.batch_id} "
                        f"como pendiente para reintento automático."
                    )
        
        return perform_enqueue

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Retorna telemetría detallada del estado actual para el visualizador HUD.
        Cumple estrictamente los campos requeridos en la especificación:
        - Greeting Batch
        - Reason
        - Cooldowns por persona
        """
        session = self.current_session
        if not session:
            return {
                "active_session": False,
                "session_id": None,
                "state": "NO_INTERACTION",
                "participants": [],
                "greeting_batch": [],
                "batch_reason": "NONE",
                "group_type": "NONE",
                "greeting_text": "",
                "is_audio_playing": self.audio_queue.is_playing(),
                "audio_current_text": self.audio_queue.get_current_text(),
                "total_greetings": self.total_greetings_emitted,
                "last_blocked_reason": self.last_blocked_reason,
                "cooldowns": {}
            }

        # Extraer nombres del lote activo o del último lote completado
        if session.active_batch:
            batch_names = [p.identity_label for p in session.active_batch.participants]
            batch_text = session.active_batch.greeting_text
            batch_type = session.active_batch.group_type.value
            batch_reason = session.active_batch.reason
        elif session.batch_history:
            last = session.batch_history[-1]
            batch_names = [p.identity_label for p in last.participants]
            batch_text = last.greeting_text
            batch_type = last.group_type.value
            batch_reason = f"{last.reason} (COMPLETED)"
        else:
            batch_names = []
            batch_text = ""
            batch_type = "NONE"
            batch_reason = self.last_batch_reason

        # Generar mapa de cooldowns por persona
        cooldown_map = self.cooldown_manager.get_cooldown_summary(list(session.participants.values()))

        return {
            "active_session": True,
            "session_id": session.session_id,
            "state": session.state.value,
            "participants": [
                {"id": p.track_id, "name": p.identity_label, "decision": p.identity_decision, "greeted": p.greeted_in_session}
                for p in session.participants.values()
            ],
            "greeting_batch": batch_names,
            "batch_reason": batch_reason,
            "group_type": batch_type,
            "greeting_text": batch_text,
            "is_audio_playing": self.audio_queue.is_playing(),
            "audio_current_text": self.audio_queue.get_current_text(),
            "total_greetings": self.total_greetings_emitted,
            "last_blocked_reason": self.last_blocked_reason,
            "cooldowns": cooldown_map
        }
