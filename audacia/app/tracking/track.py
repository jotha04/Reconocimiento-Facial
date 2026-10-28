"""
Representación de una pista temporal (Track) individual para una persona detectada.
Garantiza la separación conceptual: Track ID != Identidad de Persona.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2


class TrackState(Enum):
    """Estados del ciclo de interacción de un Track en AudacIA."""
    OBSERVED = "OBSERVED"                      # Rostro recién aparecido
    ATTENTION_CANDIDATE = "ATTENTION_CANDIDATE"# Evidencia temporal y geométrica preliminar
    ENGAGING = "ENGAGING"                      # Interacción activa confirmada frente a la cámara
    DISENGAGED = "DISENGAGED"                  # Perdió atención o abandonó el encuadre


@dataclass
class Track:
    """
    Estructura de datos que almacena el historial y estado continuo de un rostro rastreado.
    """
    track_id: int
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)

    # Geometría actual
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    center: Tuple[int, int] = (0, 0)
    width: int = 0
    height: int = 0
    landmarks: Optional[np.ndarray] = None            # shape: (5, 2)
    head_orientation_proxy: float = 0.50    # t_nose proyectado (0.50 = centro)
    orientation_frontal_score: float = 0.0 # Puntaje continuo de alineación frontal [0.0, 1.0]
    raw_detection: Optional[np.ndarray] = None
    is_complete_face: bool = True

    # Cinemática y filtros
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    # Historial de observaciones
    frames_seen: int = 1
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    age_frames: int = 1

    # Estado y Atención
    state: TrackState = TrackState.OBSERVED
    attention_score: float = 0.0
    attention_components: Dict[str, float] = field(default_factory=dict)
    frontal_attention_since: Optional[float] = None

    # Identidad y Reconocimiento Facial (Fase 5)
    identity_decision: str = "NOT_EVALUATED"  # "NOT_EVALUATED", "KNOWN", "UNKNOWN"
    identity_label: str = "NOT_EVALUATED"     # "NOT_EVALUATED", "UNKNOWN" o Nombre
    person_id: Optional[str] = None
    visitor_id: Optional[int] = None
    recognition_similarity: float = 0.0      # Similitud coseno del match [0.0, 1.0]
    recognition_eval_count: int = 0
    recognition_candidate_history: list = field(default_factory=list)
    embedding: Optional[np.ndarray] = None   # Vector 128-D de características faciales
    embedding_samples: list = field(default_factory=list)
    _last_identity_verification_time: float = 0.0
    _identity_mismatch_count: int = 0

    @property
    def dwell_time(self) -> float:
        """Tiempo continuo en segundos desde que el track fue instanciado."""
        return max(0.0, time.monotonic() - self.created_at)

    @property
    def frontal_attention_duration(self) -> float:
        """Tiempo continuo en segundos de atención frontal ininterrumpida."""
        if self.frontal_attention_since is None:
            return 0.0
        return max(0.0, time.monotonic() - self.frontal_attention_since)

    @property
    def time_since_last_seen(self) -> float:
        """Tiempo en segundos desde la última detección física confirmada."""
        return max(0.0, time.monotonic() - self.last_seen)

    @property
    def is_confirmed(self) -> bool:
        """Indica si el track ha superado el mínimo de detecciones consecutivas iniciales."""
        return self.frames_seen >= 3 or self.consecutive_hits >= 3
