"""
Gestor de Cooldowns por Persona y Sesión (CooldownManager).
Garantiza:
- Cooldown individual por persona registrada (NO global).
- Previene repetición de saludos continuos frente a la cámara.
- Cooldown para interacciones anónimas (UNKNOWN).
- Filtrado de participantes elegibles para GreetingBatch.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from audacia.app.interaction.session import Participant

logger = logging.getLogger(__name__)


class CooldownManager:
    """
    Administra los tiempos de enfriamiento individuales y de grupo.
    """

    def __init__(
        self,
        person_cooldown_seconds: float = 45.0,
        unknown_cooldown_seconds: float = 25.0,
        appearance_similarity_threshold: float = 0.65
    ):
        self.person_cooldown_seconds = person_cooldown_seconds
        self.unknown_cooldown_seconds = unknown_cooldown_seconds
        self.appearance_similarity_threshold = appearance_similarity_threshold

        # Registro histórico: person_id o nombre -> timestamp del último saludo
        self._last_greeted_persons: Dict[str, float] = {}

        # Cooldowns efímeros para desconocidos (UNKNOWN):
        # Mapeo track_id -> timestamp del último saludo
        self._unknown_track_cooldowns: Dict[int, float] = {}

        # Memoria efímera de huellas espaciales recientes de desconocidos saludados
        # Lista de dicts: [{"timestamp": float, "center_x": float, "center_y": float, "bbox_height": int, "track_id": int}]
        self._recent_unknown_footprints: List[Dict[str, float]] = []

        # Memoria efímera de apariencia facial (embeddings SFace 128-D) de desconocidos saludados
        # Lista de dicts: [{"timestamp": float, "embedding": np.ndarray, "track_id": int, "center_x": float, "center_y": float, "bbox_height": int}]
        self._recent_unknown_appearances: List[Dict[str, Any]] = []

        # Memoria unificada de huellas espaciales recientes para todos los saludos emitidos (evita repetición en el mismo lugar)
        self._recent_general_footprints: List[Dict[str, Any]] = []

        # Timestamp del último saludo a persona/grupo anónimo (mantenido para HUD/telemetría)
        self._last_unknown_greeting: float = 0.0

    def _collect_greeted_times(self, person_id: Optional[str], name: Optional[str]) -> List[float]:
        times = []
        keys = set()
        if person_id:
            keys.add(person_id)
            keys.add(person_id.lower().strip())
            keys.add(person_id.replace("_", " ").lower().strip())
        if name:
            keys.add(name)
            keys.add(name.lower().strip())
            keys.add(name.replace(" ", "_").lower().strip())
        for k in keys:
            if k in self._last_greeted_persons:
                times.append(self._last_greeted_persons[k])
        return times

    def is_person_in_cooldown(self, person_id: Optional[str], name: Optional[str]) -> bool:
        """Verifica si una persona específica está dentro de su ventana de cooldown."""
        times = self._collect_greeted_times(person_id, name)
        if not times:
            return False

        elapsed = time.monotonic() - max(times)
        return elapsed < self.person_cooldown_seconds

    def get_remaining_cooldown(self, person_id: Optional[str], name: Optional[str]) -> float:
        """Retorna los segundos restantes de cooldown para una persona (0.0 si no hay cooldown)."""
        times = self._collect_greeted_times(person_id, name)
        if not times:
            return 0.0

        elapsed = time.monotonic() - max(times)
        remaining = self.person_cooldown_seconds - elapsed
        return max(0.0, remaining)

    def get_latest_greeting_remaining_cooldown(self) -> Tuple[float, Optional[str]]:
        """Retorna (segundos_restantes, nombre_o_label) del saludo más reciente aún en cooldown."""
        now = time.monotonic()
        best_rem = 0.0
        best_who = None

        for k, ts in self._last_greeted_persons.items():
            rem = self.person_cooldown_seconds - (now - ts)
            if rem > best_rem:
                best_rem = rem
                best_who = k

        if self._last_unknown_greeting > 0:
            rem_u = self.unknown_cooldown_seconds - (now - self._last_unknown_greeting)
            if rem_u > best_rem:
                best_rem = rem_u
                best_who = "Visitante"

        return max(0.0, best_rem), best_who

    def is_unknown_in_cooldown(
        self,
        track_id: Optional[int] = None,
        visitor_id: Optional[int] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        bbox_height: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        anti_flicker_window_seconds: float = 5.0,
        spatial_proximity_threshold: float = 0.15,
        scale_tolerance_ratio: float = 0.30
    ) -> bool:
        """
        Verifica si un participante anónimo (UNKNOWN) está en cooldown.
        Distingue entre:
        - Mismo visitor_id recurrente (cooldown de 25s por visitante).
        - Mismo track_id previamente saludado (cooldown de 25s por track).
        - Memoria efímera de apariencia facial (SFace 128-D) resistente a movimiento de cámara.
        - Reaparición inmediata en la misma coordenada espacial por flickering (< 5.0s, delta_x < 0.15)
          SOLO cuando el embedding aún no está disponible.
        - Nuevo visitante independiente en posición distinta o tiempo posterior (NO bloqueado).
        """
        now = time.monotonic()

        # 1. Si no se especificaron identificadores ni coordenadas, compatibilidad hacia atrás
        if track_id is None and visitor_id is None and center_x is None:
            elapsed = now - self._last_unknown_greeting
            in_cd = elapsed < self.unknown_cooldown_seconds
            if in_cd:
                logger.info(
                    f"[FOOTPRINT CHECK]\n"
                    f"  candidate_track={track_id}\n"
                    f"  matched_footprint=GLOBAL\n"
                    f"  reason=GLOBAL_UNKNOWN_COOLDOWN"
                )
            return in_cd

        # 2a. Comprobar si este visitor_id recurrente fue saludado recientemente
        if visitor_id is not None:
            v_key = f"visitor_{visitor_id}"
            if v_key in self._unknown_track_cooldowns:
                elapsed = now - self._unknown_track_cooldowns[v_key]
                if elapsed < self.unknown_cooldown_seconds:
                    logger.info(
                        f"[FOOTPRINT CHECK]\n"
                        f"  candidate_track={track_id}\n"
                        f"  matched_visitor={visitor_id}\n"
                        f"  reason=VISITOR_COOLDOWN ({elapsed:.1f}s / {self.unknown_cooldown_seconds}s)"
                    )
                    return True

        # 2b. Comprobar si este track_id específico fue saludado recientemente
        if track_id is not None and track_id in self._unknown_track_cooldowns:
            elapsed = now - self._unknown_track_cooldowns[track_id]
            if elapsed < self.unknown_cooldown_seconds:
                logger.info(
                    f"[FOOTPRINT CHECK]\n"
                    f"  candidate_track={track_id}\n"
                    f"  matched_footprint={track_id}\n"
                    f"  reason=TRACK_COOLDOWN"
                )
                return True

        # 2c. Comprobar memoria efímera de apariencia facial (SFace 128-D)
        has_embedding_match = False
        if embedding is not None and len(self._recent_unknown_appearances) > 0:
            q_emb = np.ascontiguousarray(embedding, dtype=np.float32).flatten()
            q_norm = float(np.linalg.norm(q_emb))
            if q_norm > 1e-6:
                for rec in self._recent_unknown_appearances:
                    dt = now - rec["timestamp"]
                    if dt < self.unknown_cooldown_seconds:
                        ref_emb = rec["embedding"]
                        ref_norm = float(np.linalg.norm(ref_emb))
                        if ref_norm > 1e-6:
                            sim = float(np.dot(q_emb, ref_emb) / (q_norm * ref_norm))
                            dx_v = abs(center_x - rec["center_x"]) if center_x is not None else 0.0
                            dy_v = abs(center_y - rec["center_y"]) if center_y is not None else 0.0
                            is_same_appearance = (sim >= self.appearance_similarity_threshold) or (sim >= 0.58 and dx_v < 0.20 and dy_v < 0.20)
                            if is_same_appearance:
                                logger.info(
                                    f"[FOOTPRINT CHECK]\n"
                                    f"  candidate_track={track_id}\n"
                                    f"  matched_footprint={rec['track_id']} (APPEARANCE)\n"
                                    f"  dx={dx_v:.3f}\n"
                                    f"  dy={dy_v:.3f}\n"
                                    f"  scale_delta={sim:.3f}\n"
                                    f"  reason=APPEARANCE_COOLDOWN (Sim: {sim:.3f}, dt: {dt:.1f}s)"
                                )
                                return True

        # 3. Comprobar huella espacial reciente (anti-flickering y reingreso inmediato del mismo individuo)
        # CRÍTICO: Si ya tenemos embedding y NO coincidió con ninguna apariencia previa,
        # la persona está confirmada biométricamente como DISTINTA. NO bloquear por huella espacial
        # para garantizar que nuevos visitantes SIEMPRE sean saludados aunque se paren en el mismo sitio.
        if embedding is None and center_x is not None and center_y is not None and bbox_height is not None:
            for fp in self._recent_unknown_footprints:
                dt = now - fp["timestamp"]
                dx = abs(center_x - fp["center_x"])
                dy = abs(center_y - fp["center_y"])
                h_diff_ratio = abs(bbox_height - fp["bbox_height"]) / max(1, fp["bbox_height"]) if fp["bbox_height"] > 0 else 0

                if dt < anti_flicker_window_seconds and dx < spatial_proximity_threshold and dy < spatial_proximity_threshold and h_diff_ratio < scale_tolerance_ratio:
                    logger.info(
                        f"[FOOTPRINT CHECK]\n"
                        f"  candidate_track={track_id}\n"
                        f"  matched_footprint={fp['track_id']}\n"
                        f"  dx={dx:.3f}\n"
                        f"  dy={dy:.3f}\n"
                        f"  scale_delta={h_diff_ratio:.3f}\n"
                        f"  reason=FOOTPRINT_COOLDOWN"
                    )
                    return True

        return False

    def get_unknown_remaining_cooldown(
        self,
        track_id: Optional[int] = None,
        visitor_id: Optional[int] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        bbox_height: Optional[int] = None,
        embedding: Optional[np.ndarray] = None
    ) -> float:
        """Retorna los segundos restantes de cooldown para participantes desconocidos."""
        now = time.monotonic()

        if visitor_id is not None:
            v_key = f"visitor_{visitor_id}"
            if v_key in self._unknown_track_cooldowns:
                elapsed = now - self._unknown_track_cooldowns[v_key]
                rem = self.unknown_cooldown_seconds - elapsed
                if rem > 0:
                    return rem

        if track_id is not None and track_id in self._unknown_track_cooldowns:
            elapsed = now - self._unknown_track_cooldowns[track_id]
            return max(0.0, self.unknown_cooldown_seconds - elapsed)

        if embedding is not None and len(self._recent_unknown_appearances) > 0:
            q_emb = np.ascontiguousarray(embedding, dtype=np.float32).flatten()
            q_norm = float(np.linalg.norm(q_emb))
            if q_norm > 1e-6:
                for rec in reversed(self._recent_unknown_appearances):
                    dt = now - rec["timestamp"]
                    if dt < self.unknown_cooldown_seconds:
                        ref_emb = rec["embedding"]
                        ref_norm = float(np.linalg.norm(ref_emb))
                        if ref_norm > 1e-6:
                            sim = float(np.dot(q_emb, ref_emb) / (q_norm * ref_norm))
                            dx_v = abs(center_x - rec["center_x"]) if center_x is not None else 0.0
                            dy_v = abs(center_y - rec["center_y"]) if center_y is not None else 0.0
                            if (sim >= self.appearance_similarity_threshold) or (sim >= 0.58 and dx_v < 0.20 and dy_v < 0.20):
                                return max(0.0, self.unknown_cooldown_seconds - dt)

        if embedding is None and center_x is not None and center_y is not None and bbox_height is not None:
            for fp in reversed(self._recent_unknown_footprints):
                dt = now - fp["timestamp"]
                if dt < 5.0:
                    dx = abs(center_x - fp["center_x"])
                    dy = abs(center_y - fp["center_y"])
                    h_diff_ratio = abs(bbox_height - fp["bbox_height"]) / max(1, fp["bbox_height"]) if fp["bbox_height"] > 0 else 0
                    if dx < 0.15 and dy < 0.15 and h_diff_ratio < 0.30:
                        return max(0.0, 5.0 - dt)

        if track_id is None and visitor_id is None and center_x is None and embedding is None:
            elapsed = now - self._last_unknown_greeting
            return max(0.0, self.unknown_cooldown_seconds - elapsed)

        return 0.0

    def check_participant(self, p: Participant) -> Tuple[bool, str, float, str]:
        """
        Retorna (is_in_cooldown, status_str, remaining_s, reason).
        Aplica:
        1. Cooldown de identidad nominal si es KNOWN (por persona, 45s).
        2. Cooldown específico de desconocido si es UNKNOWN (visitor_id / track / apariencia / huella anti-flicker).
        """
        now = time.monotonic()

        # 1. Cooldown de identidad nominal si es KNOWN
        if p.is_known:
            if self.is_person_in_cooldown(p.person_id, p.identity_label):
                rem = self.get_remaining_cooldown(p.person_id, p.identity_label)
                return True, f"ACTIVE ({rem:.0f}s)", rem, "KNOWN_COOLDOWN"
            # Si es KNOWN y NO está en cooldown de identidad, queda inmediatamente habilitado para saludo
            return False, "CLEAR", 0.0, "NONE"

        # 2. Si no es conocido (UNKNOWN), verificar cooldown específico (visitor_id / track / apariencia / huella anti-flicker)
        in_cd = self.is_unknown_in_cooldown(
            track_id=p.track_id,
            visitor_id=getattr(p, 'visitor_id', None),
            center_x=p.center_x,
            center_y=p.center_y,
            bbox_height=p.bbox_height,
            embedding=p.embedding
        )
        if in_cd:
            rem = self.get_unknown_remaining_cooldown(
                track_id=p.track_id,
                visitor_id=getattr(p, 'visitor_id', None),
                center_x=p.center_x,
                center_y=p.center_y,
                bbox_height=p.bbox_height,
                embedding=p.embedding
            )
            status = f"ACTIVE ({rem:.0f}s)"
            reason = "FOOTPRINT_COOLDOWN"
            v_id = getattr(p, 'visitor_id', None)
            if v_id is not None and f"visitor_{v_id}" in self._unknown_track_cooldowns:
                reason = "VISITOR_COOLDOWN"
            elif p.track_id is not None and p.track_id in self._unknown_track_cooldowns:
                reason = "TRACK_COOLDOWN"
            elif p.embedding is not None and len(self._recent_unknown_appearances) > 0:
                reason = "APPEARANCE_COOLDOWN"
            return True, status, rem, reason

        return False, "CLEAR", 0.0, "NONE"

    def filter_eligible_for_greeting(
        self,
        participants: List[Participant]
    ) -> Tuple[List[Participant], List[Participant]]:
        """
        Separa los participantes en:
        - elegibles para formar el GreetingBatch (no en cooldown y no saludados en sesión)
        - bloqueados por cooldown
        """
        eligible: List[Participant] = []
        blocked: List[Participant] = []

        for p in participants:
            if p.greeted_in_session:
                continue

            in_cd, status, _, reason = self.check_participant(p)
            p.cooldown_status = status
            if in_cd:
                p.cooldown_reason = reason
                blocked.append(p)
            else:
                eligible.append(p)

        return eligible, blocked

    def get_cooldown_summary(self, participants: List[Participant]) -> Dict[str, str]:
        """Genera un mapa de estados de cooldown para el HUD de diagnóstico."""
        summary = {}
        for p in participants:
            label = p.identity_label if p.is_known else f"UNKNOWN (#{p.track_id})"
            _, status, _, _ = self.check_participant(p)
            summary[label] = status
        return summary

    def record_greeting(self, participants: List[Participant]):
        """Registra el timestamp del saludo para cada participante del lote."""
        now = time.monotonic()

        # Purga de huellas espaciales y de apariencia expiradas
        self._recent_unknown_footprints = [
            fp for fp in self._recent_unknown_footprints
            if (now - fp["timestamp"]) < self.unknown_cooldown_seconds
        ]
        self._recent_unknown_appearances = [
            app for app in self._recent_unknown_appearances
            if (now - app["timestamp"]) < self.unknown_cooldown_seconds
        ]
        self._recent_general_footprints = [
            fp for fp in self._recent_general_footprints
            if (now - fp["timestamp"]) < 15.0
        ]

        for p in participants:
            # Registrar huella espacial general para TODO participante saludado (KNOWN o UNKNOWN)
            if p.center_x is not None and p.center_y is not None:
                self._recent_general_footprints.append({
                    "timestamp": now,
                    "center_x": p.center_x,
                    "center_y": p.center_y,
                    "bbox_height": p.bbox_height,
                    "track_id": p.track_id,
                    "label": p.identity_label if p.is_known else "UNKNOWN"
                })

            if p.is_known:
                if p.person_id:
                    self._last_greeted_persons[p.person_id] = now
                    self._last_greeted_persons[p.person_id.lower().strip()] = now
                    self._last_greeted_persons[p.person_id.replace("_", " ").lower().strip()] = now
                if p.identity_label:
                    self._last_greeted_persons[p.identity_label] = now
                    self._last_greeted_persons[p.identity_label.lower().strip()] = now
                    self._last_greeted_persons[p.identity_label.replace(" ", "_").lower().strip()] = now
            else:
                self._unknown_track_cooldowns[p.track_id] = now
                v_id = getattr(p, "visitor_id", None)
                if v_id is not None:
                    self._unknown_track_cooldowns[f"visitor_{v_id}"] = now
                self._recent_unknown_footprints.append({
                    "timestamp": now,
                    "center_x": p.center_x,
                    "center_y": p.center_y,
                    "bbox_height": p.bbox_height,
                    "track_id": p.track_id,
                    "visitor_id": v_id
                })
                if p.embedding is not None:
                    self._recent_unknown_appearances.append({
                        "timestamp": now,
                        "embedding": np.ascontiguousarray(p.embedding, dtype=np.float32).flatten(),
                        "track_id": p.track_id,
                        "visitor_id": v_id,
                        "center_x": p.center_x,
                        "center_y": p.center_y,
                        "bbox_height": p.bbox_height
                    })
                self._last_unknown_greeting = now
                logger.info(
                    f"[FOOTPRINT]\n"
                    f"  created:\n"
                    f"  track_id={p.track_id}\n"
                    f"  visitor_id={v_id}\n"
                    f"  x={p.center_x:.3f}\n"
                    f"  y={p.center_y:.3f}\n"
                    f"  h={p.bbox_height}\n"
                    f"  timestamp={now:.3f}"
                )

        logger.info(f"Cooldown actualizado para {len(participants)} participantes.")
