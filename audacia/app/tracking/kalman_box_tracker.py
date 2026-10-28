"""
Implementación de Filtro de Kalman 2D para seguimiento suave de bounding boxes
de rostros humanos (basado en SORT cinemático pero optimizado en C++ vía cv2.KalmanFilter).
"""

from typing import Tuple
import numpy as np
import cv2


class KalmanBoxTracker:
    """
    Rastrea el centro, escala (área) y relación de aspecto de un rostro usando Kalman.
    Vector de estado: [cx, cy, s, r, vx, vy, vs]^T
    Vector de medición: [cx, cy, s, r]^T
    """

    def __init__(self, bbox: Tuple[int, int, int, int]):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = float(w * h)
        r = float(w / max(1.0, float(h)))

        # 7 estados internos, 4 medidas observables
        self.kf = cv2.KalmanFilter(7, 4)

        # Matriz de transición de estado F (velocidad constante)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Matriz de medición H
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # Covarianzas de ruido (calibradas para rostros a 30 FPS)
        self.kf.processNoiseCov = np.diag([
            1.0, 1.0, 10.0, 0.05, 5.0, 5.0, 10.0
        ]).astype(np.float32)

        self.kf.measurementNoiseCov = np.diag([
            2.0, 2.0, 20.0, 0.1
        ]).astype(np.float32)

        self.kf.errorCovPost = np.diag([
            10.0, 10.0, 100.0, 1.0, 50.0, 50.0, 100.0
        ]).astype(np.float32)

        # Inicializar estado
        self.kf.statePost = np.array([
            [cx], [cy], [s], [r], [0.0], [0.0], [0.0]
        ], dtype=np.float32)

        self.last_predicted_bbox = bbox

    def predict(self) -> Tuple[int, int, int, int]:
        """Avanza el estado del filtro de Kalman y retorna la caja estimada."""
        pred = self.kf.predict()
        cx = float(pred[0, 0])
        cy = float(pred[1, 0])
        s = max(100.0, float(pred[2, 0]))
        r = max(0.2, min(5.0, float(pred[3, 0])))

        # Reconstruir w y h desde s y r: w = sqrt(s*r), h = sqrt(s/r)
        w = np.sqrt(s * r)
        h = np.sqrt(s / r)

        x = cx - w / 2.0
        y = cy - h / 2.0

        # Amortiguar velocidades cinemáticas en predicción libre para prevenir que el recuadro
        # se achique (vs negativo) o derive erráticamente hacia una esquina cuando el rostro es cubierto.
        self.kf.statePost[4, 0] *= 0.40  # Amortiguar vx
        self.kf.statePost[5, 0] *= 0.40  # Amortiguar vy
        self.kf.statePost[6, 0] = 0.0    # Anular vs (cambio de área) para preservar escala estable

        self.last_predicted_bbox = (int(round(x)), int(round(y)), int(round(w)), int(round(h)))
        return self.last_predicted_bbox

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Corrige el filtro con una medición confirmada y retorna velocidades (vx, vy)."""
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = float(w * h)
        r = float(w / max(1.0, float(h)))

        measurement = np.array([[cx], [cy], [s], [r]], dtype=np.float32)
        corr = self.kf.correct(measurement)

        vx = float(corr[4, 0])
        vy = float(corr[5, 0])
        return vx, vy
