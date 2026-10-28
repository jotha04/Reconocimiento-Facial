"""
Módulo de estimación de movimiento de cámara para detección de estados
CAMERA_STABLE vs CAMERA_MOVING (rotación de la cabeza del robot).
Permite comparar Phase Correlation y Sparse Optical Flow para discriminar
entre movimiento global de la cámara y movimiento local de personas u objetos.
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class MotionState(Enum):
    CAMERA_STABLE = "CAMERA_STABLE"
    CAMERA_MOVING = "CAMERA_MOVING"


@dataclass
class MotionTelemetry:
    state: MotionState
    method: str
    displacement_x: float
    displacement_y: float
    motion_magnitude: float
    confidence_or_inliers: float
    calc_time_ms: float


class CameraMotionDetector:
    """
    Detecta si la cámara montada en la cabeza del robot está girando/moviéndose
    o si se encuentra estática frente al entorno.
    """

    def __init__(
        self,
        method: str = "sparse_flow",  # 'sparse_flow' o 'phase_correlate'
        motion_threshold: float = 2.0,  # Píxeles de desplazamiento para considerar movimiento activo
        min_inlier_ratio: float = 0.60,  # En sparse_flow: % de puntos que deben coincidir con la traslación global
        downsample_width: int = 320,
        downsample_height: int = 240,
        history_smoothing: int = 3
    ):
        self.method = method
        self.motion_threshold = motion_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.downsample_width = downsample_width
        self.downsample_height = downsample_height
        self.history_smoothing = history_smoothing

        self._prev_gray: Optional[np.ndarray] = None
        self._recent_states = []

        # Preparar cuadrícula fija de puntos para sparse optical flow
        # Rejilla de 8 columnas x 6 filas en resolución reducida
        xs = np.linspace(30, self.downsample_width - 30, 8, dtype=np.float32)
        ys = np.linspace(30, self.downsample_height - 30, 6, dtype=np.float32)
        grid_pts = []
        for y in ys:
            for x in xs:
                grid_pts.append([[x, y]])
        self._grid_points = np.array(grid_pts, dtype=np.float32)

    def update(self, frame: np.ndarray) -> MotionTelemetry:
        """
        Analiza el nuevo frame y determina si la cámara está estática o en movimiento.
        """
        t0 = time.perf_counter()

        # Reducir resolución para minimizar costo de CPU (inferior a 2 ms)
        small_gray = cv2.cvtColor(
            cv2.resize(frame, (self.downsample_width, self.downsample_height), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2GRAY
        )

        if self._prev_gray is None:
            self._prev_gray = small_gray
            calc_ms = (time.perf_counter() - t0) * 1000.0
            return MotionTelemetry(
                state=MotionState.CAMERA_STABLE,
                method=self.method,
                displacement_x=0.0,
                displacement_y=0.0,
                motion_magnitude=0.0,
                confidence_or_inliers=1.0,
                calc_time_ms=calc_ms
            )

        if self.method == "phase_correlate":
            # Método A: Correlación de fase en escala de grises con ventana de Hanning
            # Nota: phaseCorrelate requiere float32
            prev_f = np.float32(self._prev_gray)
            curr_f = np.float32(small_gray)
            (dx, dy), response = cv2.phaseCorrelate(prev_f, curr_f)
            mag = float(np.sqrt(dx * dx + dy * dy))

            is_moving = mag >= self.motion_threshold and response > 0.15
            inlier_val = float(response)
            dx_val, dy_val = float(dx), float(dy)

        else:  # 'sparse_flow' (Método B: Recomendado para distinguir de objetos en primer plano)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                small_gray,
                self._grid_points,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # Filtrar puntos rastreados con éxito
            valid_mask = (status == 1).reshape(-1)
            if np.sum(valid_mask) < len(self._grid_points) * 0.4:
                # Muy pocos puntos válidos (posible blur extremo durante giro violento)
                is_moving = True
                dx_val, dy_val, mag = 0.0, 0.0, 99.0
                inlier_val = 0.0
            else:
                p0 = self._grid_points[valid_mask].reshape(-1, 2)
                p1 = next_pts[valid_mask].reshape(-1, 2)
                displacements = p1 - p0  # shape: (M, 2)

                dx_med = float(np.median(displacements[:, 0]))
                dy_med = float(np.median(displacements[:, 1]))
                mag = float(np.sqrt(dx_med * dx_med + dy_med * dy_med))

                # Medir concordancia (cuántos puntos se mueven similar a la mediana global)
                diffs = np.linalg.norm(displacements - [dx_med, dy_med], axis=1)
                inliers = np.sum(diffs < 2.5)
                inlier_ratio = float(inliers / len(displacements))

                # Si el desplazamiento de la mayoría supera el umbral y los puntos concuerdan, la cámara se mueve
                is_moving = (mag >= self.motion_threshold) and (inlier_ratio >= self.min_inlier_ratio)
                dx_val, dy_val = dx_med, dy_med
                inlier_val = inlier_ratio

        self._prev_gray = small_gray

        # Suavizado temporal de estados para evitar parpadeo en transiciones
        self._recent_states.append(is_moving)
        if len(self._recent_states) > self.history_smoothing:
            self._recent_states.pop(0)

        # Si más del 50% de los frames recientes reportan movimiento, marcar MOVING
        consensus_moving = (sum(self._recent_states) / len(self._recent_states)) >= 0.5
        final_state = MotionState.CAMERA_MOVING if consensus_moving else MotionState.CAMERA_STABLE

        calc_ms = (time.perf_counter() - t0) * 1000.0
        return MotionTelemetry(
            state=final_state,
            method=self.method,
            displacement_x=round(dx_val, 2),
            displacement_y=round(dy_val, 2),
            motion_magnitude=round(mag, 2),
            confidence_or_inliers=round(inlier_val, 3),
            calc_time_ms=round(calc_ms, 2)
        )
