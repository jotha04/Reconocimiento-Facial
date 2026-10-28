"""
Motor de Atención e Interacción (Attention / Interaction Engine) de AudacIA.
Versión corregida: Implementa compuertas conjuntivas multiplicativas
(la orientación frontal y el tamaño físico actúan como compuertas obligatorias).
Evita que personas de perfil, de medio lado o pequeños parches de fondo
acumulen puntaje por mera persistencia estática.
"""

import time
import logging
from typing import List, Dict, Optional
import numpy as np

from audacia.app.tracking.track import Track, TrackState
from audacia.app.attention.config import AttentionConfig

logger = logging.getLogger(__name__)


class AttentionEngine:
    """
    Calcula el AttentionScore in [0.0, 1.0] mediante compuertas conjuntivas
    y gestiona la máquina de estados de interacción para cada track activo.
    """

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        frame_width: float = 1920.0,
        frame_height: float = 1080.0
    ):
        self.config = config or AttentionConfig()
        self.frame_width = max(1.0, float(frame_width))
        self.frame_height = max(1.0, float(frame_height))
        self._low_attention_since: Dict[int, float] = {}
        self._frontal_attention_drop_since: Dict[int, float] = {}
        self._last_telemetry_time: Dict[int, float] = {}

    def update(self, tracks: List[Track], now: Optional[float] = None) -> List[Track]:
        if now is None:
            now = time.monotonic()
        active_ids = {t.track_id for t in tracks}

        # Limpiar referencias de tracks extintos
        for tid in list(self._low_attention_since.keys()):
            if tid not in active_ids:
                del self._low_attention_since[tid]

        for tid in list(self._frontal_attention_drop_since.keys()):
            if tid not in active_ids:
                del self._frontal_attention_drop_since[tid]
        
        for tid in list(self._last_telemetry_time.keys()):
            if tid not in active_ids:
                del self._last_telemetry_time[tid]

        for track in tracks:
            # 1. Calcular señales individuales en [0.0, 1.0]
            s_orient = self._compute_orientation_score(track)
            s_size = self._compute_size_score(track)
            s_pers = self._compute_persistence_score(track)
            s_pos = self._compute_position_score(track)
            s_stab = self._compute_stability_score(track)

            # 2. Ponderación Conjuntiva con Compuerta de Orientación y Tamaño
            # Si la persona no está mirando a la cámara (s_orient bajo) o está muy lejos (s_size bajo),
            # la persistencia NO puede compensar. La atención global se colapsa.
            base_attention = (
                0.35 * s_pers +
                0.30 * s_size +
                0.20 * s_pos +
                0.15 * s_stab
            )
            # Modulación conjuntiva: multiplicada por la orientación frontal
            composite_score = float(s_orient * base_attention)
            composite_score = max(0.0, min(1.0, composite_score))

            track.attention_score = round(composite_score, 2)
            track.attention_components = {
                "persistence": round(s_pers, 2),
                "size": round(s_size, 2),
                "position": round(s_pos, 2),
                "orientation": round(s_orient, 2),
                "stability": round(s_stab, 2)
            }

            # 3. Transiciones de la Máquina de Estados con Compuertas Estrictas
            self._update_track_state(track, now, s_orient, s_size)

            # Telemetry for diagnostic every 0.5s for non-ENGAGING tracks
            if track.state != TrackState.ENGAGING:
                last_log = self._last_telemetry_time.get(track.track_id, 0.0)
                if now - last_log >= 0.5:
                    self._last_telemetry_time[track.track_id] = now
                    logger.debug(
                        f"TELEMETRY Track #{track.track_id} | "
                        f"State: {track.state.name} | "
                        f"Score: {track.attention_score:.2f} | "
                        f"Persist: {s_pers:.2f} | Size: {s_size:.2f} | Pos: {s_pos:.2f} | "
                        f"Orient: {s_orient:.2f} | h: {track.height}px | "
                        f"FrontalSince: {(now - track.frontal_attention_since) if track.frontal_attention_since else 0.0:.2f}s | "
                        f"Dwell: {track.dwell_time:.1f}s"
                    )

        return tracks

    def _compute_orientation_score(self, track: Track) -> float:
        """
        Retorna la puntuación de orientación frontal continua [0.0, 1.0].
        Usa el puntaje proyectivo ya calculado por YuNetDetector si está disponible,
        o lo recalcula sobre el proxy de la nariz.
        """
        if hasattr(track, 'orientation_frontal_score') and track.orientation_frontal_score > 0:
            return float(track.orientation_frontal_score)

        # Fallback si no está precalculado: t_nose proyectado
        t_nose = track.head_orientation_proxy
        dev = abs(t_nose - 0.50)
        score = np.exp(-(dev ** 2) / (2.0 * (0.12 ** 2)))
        return float(np.clip(score, 0.0, 1.0))

    def _compute_size_score(self, track: Track) -> float:
        """Evalúa la escala del rostro en píxeles. Rostros pequeños (< 70px) reciben 0.0."""
        h = track.height
        min_h = self.config.min_face_height         # ej. 65 px
        opt_min = self.config.optimal_face_height_min # ej. 100 px
        opt_max = self.config.optimal_face_height_max # ej. 260 px
        max_h = self.config.max_face_height         # ej. 380 px

        if h < min_h:
            # Demasiado pequeño o ruido: puntaje nulo
            return 0.0
        elif h < opt_min:
            # Transición hacia zona óptima (0.2 a 0.85)
            return float(0.2 + (h - min_h) / max(1, opt_min - min_h) * 0.65)
        elif h <= opt_max:
            # Zona óptima de interacción frontal (1.0m - 1.8m)
            return 1.0
        elif h <= max_h:
            return float(0.90 - (h - opt_max) / max(1, max_h - opt_max) * 0.25)
        else:
            return 0.60

    def _compute_persistence_score(self, track: Track) -> float:
        """Evalúa la persistencia continua en segundos."""
        dwell = track.dwell_time
        t_cand = self.config.dwell_candidate_seconds
        t_eng = self.config.dwell_engaging_seconds

        if dwell < 0.2:
            return 0.05
        elif dwell < t_cand:
            return float(0.05 + (dwell - 0.2) / max(0.01, t_cand - 0.2) * 0.40)
        elif dwell < t_eng:
            return float(0.45 + (dwell - t_cand) / max(0.01, t_eng - t_cand) * 0.45)
        else:
            excess = dwell - t_eng
            return float(min(1.0, 0.90 + 0.10 * (1.0 - np.exp(-excess / 1.5))))

    def _compute_position_score(self, track: Track) -> float:
        """Evalúa la posición horizontal en el encuadre. Se analiza el 100% de la sección de la cámara."""
        z_min = getattr(self.config, 'central_zone_min_x', 0.0)
        z_max = getattr(self.config, 'central_zone_max_x', 1.0)
        if z_min <= 0.05 and z_max >= 0.95:
            return 1.0

        fw = getattr(self, "frame_width", 1920.0) or 1920.0
        if track.center and track.center[0] > fw:
            fw = max(fw, float(track.center[0]) * 1.1)
            self.frame_width = fw
        norm_cx = max(0.0, min(1.0, track.center[0] / fw))

        if z_min <= norm_cx <= z_max:
            return 1.0
        elif norm_cx < z_min:
            return float(max(0.1, norm_cx / max(0.01, z_min)))
        else:
            dist_right = 1.0 - norm_cx
            max_dist = 1.0 - z_max
            return float(max(0.1, dist_right / max(0.01, max_dist)))

    def _compute_stability_score(self, track: Track) -> float:
        """Evalúa continuidad temporal y suavidad de movimiento."""
        total_events = track.frames_seen + track.consecutive_misses
        hit_ratio = float(track.frames_seen / max(1, total_events))

        speed = float(np.sqrt(track.velocity_x ** 2 + track.velocity_y ** 2))
        vel_penalty = 1.0 if speed < 35.0 else max(0.4, 1.0 - (speed - 35.0) / 80.0)

        return float(np.clip(hit_ratio * vel_penalty, 0.1, 1.0))

    def _update_track_state(self, track: Track, now: float, s_orient: float, s_size: float):
        """
        Máquina de Estados de Interacción con Compuertas Estrictas:
        - Para ser CANDIDATE: Requiere tamaño físico real (h >= 65px) Y orientación razonable (s_orient >= 0.50).
        - Para ser ENGAGING: Requiere tamaño óptimo (h >= 75px), ORIENTACIÓN FRONTAL ESTRICTA (s_orient >= 0.70),
          persistencia continua (dwell >= dwell_engaging_seconds) y Score >= engaging_score_threshold.
        """
        score = track.attention_score
        dwell = track.dwell_time
        current_state = track.state
        tid = track.track_id
        h = track.height

        t_cand = self.config.dwell_candidate_seconds
        t_eng = self.config.dwell_engaging_seconds
        th_cand = self.config.candidate_score_threshold   # ej. 0.35
        th_eng = self.config.engaging_score_threshold     # ej. 0.60
        grace = self.config.disengage_grace_seconds

        # Compuerta estricta: si mira hacia otro lado (s_orient < 0.45) o es demasiado pequeño
        has_min_geometry = (h >= self.config.min_face_height) and (s_orient >= 0.45)
        has_engaging_geometry = (h >= self.config.min_face_height) and (s_orient >= self.config.min_orientation_for_engaging)

        # Manejo de atención frontal continua con tolerancia temporal (gracia de 250-300ms a caídas breves)
        frontal_grace = getattr(self.config, 'frontal_orientation_grace_seconds', 0.30)

        if has_engaging_geometry:
            # Si estaba en período de gracia por caída transitoria de orientación, se cancela/recupera
            if tid in self._frontal_attention_drop_since:
                del self._frontal_attention_drop_since[tid]

            if getattr(track, 'frontal_attention_since', None) is None:
                track.frontal_attention_since = now
                logger.info(f"[DIAGNOSTICS] Track #{tid} inicia frontal_attention_since en {now:.3f} (h={h}, s_orient={s_orient:.2f})")
        else:
            if getattr(track, 'frontal_attention_since', None) is not None:
                if tid not in self._frontal_attention_drop_since:
                    self._frontal_attention_drop_since[tid] = now
                elif (now - self._frontal_attention_drop_since[tid]) >= frontal_grace:
                    drop_dur = now - self._frontal_attention_drop_since[tid]
                    logger.info(f"[DIAGNOSTICS] Track #{tid} pierde frontal_attention_since tras {drop_dur:.3f}s de caída sostenida (h={h}, s_orient={s_orient:.2f})")
                    track.frontal_attention_since = None
                    del self._frontal_attention_drop_since[tid]
            else:
                if tid in self._frontal_attention_drop_since:
                    del self._frontal_attention_drop_since[tid]

        frontal_attention_duration = (
            (now - track.frontal_attention_since)
            if track.frontal_attention_since is not None
            else 0.0
        )

        # BUG-008 fix: No aplicar el filtro de atención baja genérico a tracks en ENGAGING
        # ya que ENGAGING tiene su propio umbral más estricto (0.60 vs 0.45)
        if current_state != TrackState.ENGAGING:
            is_low_attention = (score < th_cand) or (not has_min_geometry)
            if is_low_attention:
                if tid not in self._low_attention_since:
                    self._low_attention_since[tid] = now
            else:
                if tid in self._low_attention_since:
                    del self._low_attention_since[tid]

        # 1. Estado OBSERVED
        if current_state == TrackState.OBSERVED:
            # Solo transiciona a CANDIDATE si cumple los requisitos geométricos y de persistencia
            if dwell >= t_cand and score >= th_cand and has_min_geometry:
                track.state = TrackState.ATTENTION_CANDIDATE
                logger.info(f"[DIAGNOSTICS] Track #{tid} transicionó a ATTENTION_CANDIDATE (Score: {score:.2f})")

        # 2. Estado ATTENTION_CANDIDATE
        elif current_state == TrackState.ATTENTION_CANDIDATE:
            # BUG-006 fix: Requiere atención frontal continua suficiente (frontal_attention_duration >= t_eng)
            if (
                frontal_attention_duration >= t_eng
                and score >= th_eng
                and has_engaging_geometry
            ):
                track.state = TrackState.ENGAGING
                logger.info(
                    f"Track #{tid} transicionó a ENGAGING "
                    f"(Score: {score:.2f}, Atención: {frontal_attention_duration:.2f}s, Dwell: {dwell:.1f}s, Orient: {s_orient:.2f})"
                )
            elif tid in self._low_attention_since and (now - self._low_attention_since[tid]) >= grace:
                track.state = TrackState.DISENGAGED
                logger.debug(f"Track #{tid} cayó a DISENGAGED por atención insuficiente.")

        # 3. Estado ENGAGING
        elif current_state == TrackState.ENGAGING:
            # BUG-008 fix: Gestión aislada e inequívoca del temporizador de desenganche
            lost_engagement = (s_orient < self.config.min_orientation_for_engaging) or (score < th_cand) or (not has_min_geometry)
            if lost_engagement:
                if tid not in self._low_attention_since:
                    self._low_attention_since[tid] = now
                elif (now - self._low_attention_since[tid]) >= grace:
                    track.state = TrackState.DISENGAGED
                    track.frontal_attention_since = None
                    if tid in self._frontal_attention_drop_since:
                        del self._frontal_attention_drop_since[tid]
                    del self._low_attention_since[tid]
                    logger.info(f"Track #{tid} desenganchado (DISENGAGED por orientación/atención).")
            else:
                if tid in self._low_attention_since:
                    del self._low_attention_since[tid]

        # 4. Estado DISENGAGED
        elif current_state == TrackState.DISENGAGED:
            # Para recuperarse de DISENGAGED a ATTENTION_CANDIDATE, requiere recuperar orientación frontal (>= min_orientation_for_engaging)
            if score >= th_cand and s_orient >= self.config.min_orientation_for_engaging and has_min_geometry:
                track.state = TrackState.ATTENTION_CANDIDATE
                logger.debug(f"Track #{tid} recuperado a ATTENTION_CANDIDATE.")
