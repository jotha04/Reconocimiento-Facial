"""
Modelado del ciclo de vida de una sesión de interacción (Interaction Session)
y lotes de saludo (GreetingBatch) para AudacIA.
Garantiza las separaciones conceptuales clave:
- Track ID != Identity
- Identity != Interaction Session
- Persona visible != Participante
- Sesión Visual != Greeting Batch
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


class InteractionState(Enum):
    """Estados del ciclo de vida global de interacción del robot."""
    NO_INTERACTION = "NO_INTERACTION"              # Sin participantes en interacción activa
    CANDIDATE_GROUP = "CANDIDATE_GROUP"            # Uno o más tracks entran a ENGAGING
    GROUP_STABILIZING = "GROUP_STABILIZING"        # Ventana breve para estabilizar composición
    IDENTITIES_RESOLVING = "IDENTITIES_RESOLVING"  # Esperando consolidación de identidades pendientes
    GROUP_CONFIRMED = "GROUP_CONFIRMED"            # Grupo cerrado y verificado
    GREETING_QUEUED = "GREETING_QUEUED"            # Saludo enviado a la cola de audio
    GREETING_PLAYING = "GREETING_PLAYING"          # Audio en reproducción activa
    ACTIVE_ENGAGEMENT = "ACTIVE_ENGAGEMENT"        # Grupo interactuando tras el saludo
    COOLDOWN = "COOLDOWN"                          # Finalizando sesión y asentando cooldowns


class GroupType(Enum):
    """Clasificación de la composición de un lote de saludo."""
    SINGLE_KNOWN = "SINGLE_KNOWN"          # 1 persona registrada (ej. "Jonathan")
    MULTIPLE_KNOWN = "MULTIPLE_KNOWN"      # N personas, todas registradas (ej. "Jonathan y María")
    SINGLE_UNKNOWN = "SINGLE_UNKNOWN"      # 1 persona no registrada
    MULTIPLE_UNKNOWN = "MULTIPLE_UNKNOWN"  # 2 o más personas, ninguna registrada
    MIXED = "MIXED"                        # Grupo mixto: al menos 1 conocida y 1 desconocida


@dataclass
class Participant:
    """Representa a un individuo calificado como participante en la sesión."""
    track_id: int
    identity_label: str
    identity_decision: str      # "KNOWN", "UNKNOWN", "NOT_EVALUATED"
    person_id: Optional[str] = None
    visitor_id: Optional[int] = None
    recognition_similarity: float = 0.0
    embedding: Optional[np.ndarray] = None
    joined_at: float = field(default_factory=time.monotonic)

    # Propiedades espaciales efímeras (para deduplicación y tracking)
    center_x: float = 0.50
    center_y: float = 0.50
    bbox_width: int = 0
    bbox_height: int = 0

    # Temporal tracking: when this participant was last seen in a frame
    last_seen_at: float = field(default_factory=time.monotonic)

    # Anti-flicker: if this participant was consumed by a reconnection, mark it
    reconnected_to: Optional[int] = None  # track_id of the new participant that absorbed this one

    # Estado de saludo y cooldown
    greeted_in_session: bool = False
    greeted_at: Optional[float] = None
    cooldown_status: str = "CLEAR"  # "CLEAR", "ACTIVE", "BLOCKED"

    @property
    def is_known(self) -> bool:
        return self.identity_decision == "KNOWN" and self.identity_label not in ("UNKNOWN", "NOT_EVALUATED")

    @property
    def is_unknown(self) -> bool:
        return self.identity_decision == "UNKNOWN"

    @property
    def is_pending(self) -> bool:
        return self.identity_decision == "NOT_EVALUATED"


@dataclass
class GreetingBatch:
    """
    Representa el conjunto específico de personas que deben recibir un saludo en este turno.
    Permite saludar a nuevos participantes sin repetir a quienes ya fueron saludados.
    """
    batch_id: int
    participants: List[Participant]
    group_type: GroupType
    greeting_text: str
    reason: str = "INITIAL_GROUP"   # "INITIAL_GROUP", "NEW_PARTICIPANTS", "LATE_JOINER"
    created_at: float = field(default_factory=time.monotonic)
    queued_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    is_completed: bool = False
    was_force_recovered: bool = False


@dataclass
class InteractionSession:
    """
    Gestiona la sesión visual continua con el grupo presente y sus lotes de saludo sucesivos.
    """
    session_id: int
    created_at: float = field(default_factory=time.monotonic)
    last_active_at: float = field(default_factory=time.monotonic)
    state: InteractionState = InteractionState.NO_INTERACTION

    # Todos los participantes actualmente en la sesión visual (track_id -> Participant)
    participants: Dict[int, Participant] = field(default_factory=dict)

    # Lote de saludo actual, lote pendiente de encolado y su historial
    active_batch: Optional[GreetingBatch] = None
    pending_batch: Optional[GreetingBatch] = None
    batch_history: List[GreetingBatch] = field(default_factory=list)
    _next_batch_id: int = 1

    # Cronómetros de control
    stabilization_start_time: Optional[float] = None
    resolution_start_time: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.created_at)

    @property
    def idle_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.last_active_at)

    def get_ungreeted_participants(self) -> List[Participant]:
        """Retorna los participantes presentes que aún no han sido saludados en esta sesión."""
        return [p for p in self.participants.values() if not p.greeted_in_session]

    def mark_batch_completed(self, batch: GreetingBatch):
        """Marca a los participantes del lote como saludados (idempotente contra carreras)."""
        if batch.is_completed:
            return
        now = time.monotonic()
        for p in batch.participants:
            if p.track_id in self.participants:
                self.participants[p.track_id].greeted_in_session = True
                self.participants[p.track_id].greeted_at = now
        batch.is_completed = True
        batch.completed_at = now
        if batch not in self.batch_history:
            self.batch_history.append(batch)
        if self.active_batch is batch:
            self.active_batch = None
        if self.pending_batch is batch:
            self.pending_batch = None
