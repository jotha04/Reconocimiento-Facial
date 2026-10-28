"""
Resolución y Estabilización de Grupos de Interacción (GroupResolver).
Garantiza:
1. Filtro estricto: Solo tracks en ENGAGING califican como participantes.
2. Ventana de estabilización temporal para incorporar participantes simultáneos.
3. Espera obligatoria hasta que las identidades estén consolidadas (KNOWN o UNKNOWN)
   y no en NOT_EVALUATED (pending).
"""

import time
import logging
from typing import List, Tuple, Dict, Optional, Any

from audacia.app.tracking.track import Track, TrackState
from audacia.app.interaction.session import Participant
from audacia.app.attention.config import InteractionConfig

logger = logging.getLogger(__name__)


class GroupResolver:
    """
    Evalúa los tracks del visualizador y resuelve el conjunto estable de participantes.
    """

    def __init__(
        self,
        config: InteractionConfig,
        attention_config: Optional[Any] = None,
        frame_width: float = 1920.0,
        frame_height: float = 1080.0
    ):
        self.config = config
        self.attention_config = attention_config
        self.frame_width = max(1.0, float(frame_width))
        self.frame_height = max(1.0, float(frame_height))

    def extract_engaging_candidates(self, tracks: List[Track]) -> List[Participant]:
        """
        Filtra únicamente los tracks que han alcanzado el estado ENGAGING.
        Los tracks en OBSERVED, ATTENTION_CANDIDATE o DISENGAGED quedan excluidos.
        """
        candidates: List[Participant] = []
        fw = getattr(self, "frame_width", 1920.0) or 1920.0
        fh = getattr(self, "frame_height", 1080.0) or 1080.0

        for t in tracks:
            if t.state == TrackState.ENGAGING:
                if t.center and t.center[0] > fw:
                    fw = max(fw, float(t.center[0]) * 1.1)
                    self.frame_width = fw
                if t.center and t.center[1] > fh:
                    fh = max(fh, float(t.center[1]) * 1.1)
                    self.frame_height = fh
                cx_norm = max(0.0, min(1.0, t.center[0] / fw)) if t.center and t.center[0] > 0 else 0.50
                cy_norm = max(0.0, min(1.0, t.center[1] / fh)) if t.center and t.center[1] > 0 else 0.50
                candidates.append(
                    Participant(
                        track_id=t.track_id,
                        identity_label=t.identity_label,
                        identity_decision=t.identity_decision,
                        person_id=getattr(t, "person_id", None),
                        visitor_id=getattr(t, "visitor_id", None),
                        recognition_similarity=t.recognition_similarity,
                        embedding=t.embedding,
                        center_x=cx_norm,
                        center_y=cy_norm,
                        bbox_width=t.width,
                        bbox_height=t.height
                    )
                )
        return candidates

    def extract_potential_companions(self, tracks: Optional[List[Track]]) -> List[Track]:
        """
        Identifica acompañantes en potencial formación de grupo:
        1. Tracks en ATTENTION_CANDIDATE con atención frontal activa.
        2. Tracks en DISENGAGED que demuestren presencia física estable (dwell >= dwell_candidate_seconds,
           h >= min_face_height, orientación frontal razonable s_orient >= 0.50 y sin movimiento
           descontrolado de transeúnte speed < 35.0 px/frame).
        Excluye transeúntes (OBSERVED, movimiento de perfil o velocidad alta de transeúnte).
        """
        if not tracks:
            return []
        companions: List[Track] = []
        min_h = getattr(self.attention_config, 'min_face_height', 70) if self.attention_config else 70
        t_cand = getattr(self.attention_config, 'dwell_candidate_seconds', 0.50) if self.attention_config else 0.50

        for t in tracks:
            if t.state == TrackState.OBSERVED:
                continue

            # Señal de velocidad kinemática del track existente (px/frame)
            speed = (t.velocity_x ** 2 + t.velocity_y ** 2) ** 0.5 if hasattr(t, 'velocity_x') else 0.0

            if t.state in (TrackState.ATTENTION_CANDIDATE, TrackState.DISENGAGED):
                # Un track es companion si:
                # 1. Ya tiene atención frontal (criterio original para CANDIDATE)
                has_attention = (t.orientation_frontal_score >= 0.60 and t.frontal_attention_duration > 0 and speed < 35.0)
                
                # 2. O demuestra presencia física estable (criterio relajado de presencia):
                # - dwell_time >= dwell_candidate_seconds (no es transitorio)
                # - height >= min_face_height (tamaño facial válido)
                # - orientation_frontal_score >= 0.50 (orientado razonablemente al dispositivo)
                # - speed < 35.0 (sin velocidad de transeúnte lateral)
                is_physically_stable = (
                    t.dwell_time >= t_cand
                    and t.height >= min_h
                    and t.orientation_frontal_score >= 0.50
                    and speed < 35.0
                )
                
                if has_attention or is_physically_stable:
                    companions.append(t)
        return companions

    def are_identities_resolved(self, participants: List[Participant]) -> bool:
        """
        Verifica que ningún participante esté en estado 'NOT_EVALUATED' (pendiente).
        Todos deben haber consolidado a KNOWN o UNKNOWN.
        """
        if not participants:
            return False
        for p in participants:
            if p.is_pending:
                return False
        return True

    def check_stabilization(
        self,
        current_candidates: List[Participant],
        tracked_participants: Dict[int, Participant],
        stabilization_start_time: Optional[float],
        resolution_start_time: Optional[float],
        all_tracks: Optional[List[Track]] = None
    ) -> Tuple[bool, bool, Optional[float], Optional[float], str]:
        """
        Verifica el estado de estabilización del grupo candidato:
        Retorna:
          (is_ready, is_timeout, new_stab_time, new_res_time, status_reason)
        """
        now = time.monotonic()

        if not current_candidates:
            return False, False, None, None, "Sin participantes en ENGAGING"

        # 1. Iniciar o actualizar ventana de estabilización
        if stabilization_start_time is None:
            stabilization_start_time = now

        elapsed_stab = now - stabilization_start_time
        # Si hay candidatos en UNKNOWN o pendientes de evaluación, extender ventana hasta 1.50s
        # para permitir que el segundo participante sea reconocido o rescatado antes de disparar el saludo
        stab_threshold = self.config.group_stabilization_seconds
        if any(not c.is_known for c in current_candidates):
            stab_threshold = max(stab_threshold, 1.50)

        if elapsed_stab < stab_threshold:
            rem = stab_threshold - elapsed_stab
            return (
                False, False, stabilization_start_time, resolution_start_time,
                f"Estabilizando grupo ({rem:.2f}s restantes, {len(current_candidates)} candidatos)"
            )

        # 2. Ventana de formación de grupo: si hay acompañantes prometedores en ATTENTION_CANDIDATE
        # mantener la ventana abierta hasta group_formation_max_seconds (p. ej. 1.5s)
        max_formation = getattr(self.config, 'group_formation_max_seconds', 1.50)
        engaging_ids = {c.track_id for c in current_candidates}
        potential_companions = [
            t for t in self.extract_potential_companions(all_tracks)
            if t.track_id not in engaging_ids
        ]

        if potential_companions and elapsed_stab < max_formation:
            rem = max_formation - elapsed_stab
            c_ids = [t.track_id for t in potential_companions]
            return (
                False, False, stabilization_start_time, resolution_start_time,
                f"Esperando formación de grupo con acompañante(s) {c_ids} ({rem:.2f}s restantes)"
            )

        # 3. Ventana de estabilización cumplida. Verificar consolidación de identidades
        if resolution_start_time is None:
            resolution_start_time = now

        elapsed_res = now - resolution_start_time
        all_resolved = self.are_identities_resolved(current_candidates)

        if not all_resolved:
            if elapsed_res < self.config.identity_resolution_timeout_seconds:
                rem = self.config.identity_resolution_timeout_seconds - elapsed_res
                return (
                    False, False, stabilization_start_time, resolution_start_time,
                    f"Esperando consolidación de identidades ({rem:.1f}s restantes)"
                )
            else:
                # Timeout de resolución: Tratar pendientes como UNKNOWN por seguridad
                logger.warning(
                    f"Timeout ({self.config.identity_resolution_timeout_seconds}s) alcanzado esperando identidades. "
                    "Participantes pendientes consolidados como UNKNOWN."
                )
                for p in current_candidates:
                    if p.is_pending:
                        p.identity_decision = "UNKNOWN"
                        p.identity_label = "UNKNOWN"

        # 4. Grupo estabilizado e identidades resueltas
        return True, False, stabilization_start_time, resolution_start_time, "Grupo estabilizado y listo"
