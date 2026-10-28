"""
Módulo de detección facial basado en OpenCV YuNet (libfacedetection).
Actualizado con métrica proyectiva de orientación de cabeza (Eye-Vector Projection)
para discriminar con precisión entre mirada frontal directa y rostros de perfil / medio lado.
"""

import time
import os
import threading
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from audacia.app.core.paths import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Representa una detección de rostro individual devuelta por YuNet."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: np.ndarray            # shape: (5, 2) -> [right_eye, left_eye, nose, right_mouth, left_mouth]
    area: int
    head_orientation_proxy: float    # Posición proyectada de la nariz respecto al eje interocular (0.50 = frontal)
    orientation_frontal_score: float # Puntaje [0.0, 1.0] de alineación frontal
    raw_detection: Optional[np.ndarray] = None # Vector nativo de 15 floats de YuNet para SFace alignCrop
    is_complete_face: bool = True     # Indica si el rostro está completo (sin cortes de bordes ni mutilaciones)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


def compute_projective_orientation(
    landmarks: np.ndarray, bbox_w: int, bbox_h: int
) -> Tuple[float, float]:
    """
    Calcula la orientación proyectiva de la cabeza usando los 5 puntos clave faciales.
    Proyecta la posición de la nariz sobre el vector director entre los dos ojos (E_l - E_r).
    
    Propiedades físicas:
    - Cara de frente hacia la cámara: La nariz cae en t_nose in [0.42, 0.58].
    - Cara girada a medio lado (~30-45°): t_nose < 0.32 o t_nose > 0.68.
    - Cara de perfil completo / mirando a otro lado: t_nose < 0.15 o t_nose > 0.85.
    
    Retorna:
    - t_nose: posición relativa (0.50 = centro exacto).
    - frontal_score: valor continuo [0.0, 1.0] que cae a ~0 en cuanto la persona gira a medio lado.
    """
    r_eye = landmarks[0]
    l_eye = landmarks[1]
    nose = landmarks[2]

    v_eyes = l_eye - r_eye
    l_eyes = float(np.linalg.norm(v_eyes))
    if l_eyes < 6.0:
        return 0.50, 0.10

    # Vector desde ojo derecho a nariz
    v_rn = nose - r_eye
    # Proyección normalizada de la nariz sobre el eje ocular
    t_nose = float(np.dot(v_rn, v_eyes) / (l_eyes ** 2))

    # Vector desde ojo derecho al centro de la boca
    mouth_center = (landmarks[3] + landmarks[4]) / 2.0
    v_rm = mouth_center - r_eye
    # Proyección normalizada de la boca sobre el eje ocular
    t_mouth = float(np.dot(v_rm, v_eyes) / (l_eyes ** 2))

    # Proporción de la separación interocular respecto al ancho de la caja
    eye_span_ratio = l_eyes / max(1.0, float(bbox_w))

    # 1. Simetría Yaw (giro horizontal izquierda/derecha)
    dev_nose = abs(t_nose - 0.50)
    dev_mouth = abs(t_mouth - 0.50)
    sym_dev = min(dev_nose, dev_mouth)
    t_used = t_nose if dev_nose < dev_mouth else t_mouth

    sigma_yaw = 0.12
    score_yaw = np.exp(-(sym_dev ** 2) / (2.0 * (sigma_yaw ** 2)))

    # 2. Inclinación Pitch vertical (mirando al frente vs agachado mirando al celular o al piso)
    eye_y_avg = (r_eye[1] + l_eye[1]) / 2.0
    mouth_y_avg = (landmarks[3][1] + landmarks[4][1]) / 2.0
    nose_y = landmarks[2][1]

    d_en = max(1.0, float(nose_y - eye_y_avg))
    d_nm = max(1.0, float(mouth_y_avg - nose_y))

    # En un rostro frontal mirando a la cámara, d_en y d_nm son proporcionados (ratio ~0.75 a 1.35).
    # Si la persona agacha la cabeza mirando al celular/piso:
    # La nariz desciende en proyección y se pega a la boca (d_nm colapsa), haciendo que d_en / d_nm se dispare a > 2.2.
    pitch_ratio = d_en / d_nm
    log_pitch_dev = abs(float(np.log2(max(0.1, pitch_ratio))))
    sigma_pitch = 0.70
    score_pitch = np.exp(-(log_pitch_dev ** 2) / (2.0 * (sigma_pitch ** 2)))

    # 3. Penalización si la separación interocular colapsa (escorzo extremo)
    if eye_span_ratio < 0.20:
        span_mult = max(0.05, eye_span_ratio / 0.20)
    elif eye_span_ratio > 0.68:
        span_mult = max(0.2, (0.75 - eye_span_ratio) / 0.07)
    else:
        span_mult = 1.0

    frontal_score = float(np.clip(score_yaw * score_pitch * span_mult, 0.0, 1.0))
    return round(t_used, 2), round(frontal_score, 3)


class YuNetDetector:
    """
    Detector facial de alta velocidad utilizando el modelo ONNX oficial de YuNet
    ejecutado a través del backend DNN nativo de OpenCV.
    """

    def __init__(
        self,
        model_path: str = "models/face_detection_yunet_2023mar.onnx",
        input_size: Tuple[int, int] = (1280, 720),
        score_threshold: float = 0.65,
        nms_threshold: float = 0.30,
        min_face_size: int = 55,  # Descarta parches pequeños de ruido y falsos positivos
        max_inference_dimension: Optional[int] = 1280
    ):
        self.model_path = resolve_path(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo YuNet en la ruta '{self.model_path}'."
            )
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        self.max_inference_dimension = max_inference_dimension

        logger.info(
            f"Inicializando YuNet: modelo={model_path}, input_size={input_size}, "
            f"score_th={score_threshold}, nms_th={nms_threshold}, min_face_size={min_face_size}px, "
            f"max_inf_dim={max_inference_dimension}"
        )

        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self.input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

        self._current_input_size = self.input_size
        self._lock = threading.Lock()

    def set_min_face_size(self, min_size: int):
        self.min_face_size = max(0, min_size)

    def detect(self, frame: np.ndarray) -> Tuple[List[FaceDetection], float]:
        """
        Ejecuta inferencia y extrae landmarks y orientación proyectiva.
        Si la resolución supera max_inference_dimension (ej. 1920x1080), optimiza la inferencia
        escalando dinámicamente y proyectando las detecciones a coordenadas 1:1 nativas.
        """
        h, w = frame.shape[:2]
        scale_x = 1.0
        scale_y = 1.0
        inference_frame = frame

        if self.max_inference_dimension and max(w, h) > self.max_inference_dimension:
            scale = self.max_inference_dimension / float(max(w, h))
            target_w = int(round(w * scale))
            target_h = int(round(h * scale))
            target_w -= (target_w % 2)
            target_h -= (target_h % 2)
            scale_x = float(w) / float(target_w)
            scale_y = float(h) / float(target_h)
            inference_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            inf_h, inf_w = target_h, target_w
        else:
            inf_h, inf_w = h, w

        with self._lock:
            if (inf_w, inf_h) != self._current_input_size:
                self.detector.setInputSize((inf_w, inf_h))
                self._current_input_size = (inf_w, inf_h)

            t_start = time.perf_counter()
            ret, faces = self.detector.detect(inference_frame)
            infer_ms = (time.perf_counter() - t_start) * 1000.0

        detections: List[FaceDetection] = []
        if faces is None or ret == 0:
            return detections, infer_ms

        for raw in faces:
            score = float(raw[14])
            if score < self.score_threshold:
                continue

            bx = int(max(0, raw[0] * scale_x))
            by = int(max(0, raw[1] * scale_y))
            bw = int(min(w - bx, raw[2] * scale_x))
            bh = int(min(h - by, raw[3] * scale_y))

            # Filtro de tamaño mínimo físico para descartar ruido y fondo lejano en coordenadas nativas
            if self.min_face_size > 0 and (bw < self.min_face_size or bh < self.min_face_size):
                continue

            # Filtro de relación de aspecto antropomórfica amplio (tolera inclinación y giros naturales)
            aspect = bw / float(bh)
            if aspect < 0.50 or aspect > 1.35:
                continue

            landmarks = np.array([
                [raw[4] * scale_x, raw[5] * scale_y],    # Ojo derecho
                [raw[6] * scale_x, raw[7] * scale_y],    # Ojo izquierdo
                [raw[8] * scale_x, raw[9] * scale_y],    # Nariz
                [raw[10] * scale_x, raw[11] * scale_y],  # Comisura derecha
                [raw[12] * scale_x, raw[13] * scale_y],  # Comisura izquierda
            ], dtype=np.float32)

            # 1. Separación interocular mínima (descarta detecciones degeneradas)
            v_eyes = landmarks[1] - landmarks[0]
            l_eyes = float(np.linalg.norm(v_eyes))
            if l_eyes < 0.15 * bw:
                continue
                
            # 2. Inclinación interocular máxima (evitar cosas verticales)
            if abs(landmarks[1][1] - landmarks[0][1]) > l_eyes * 1.5:
                continue

            # 3. Distancia mínima vertical ojos -> boca
            eye_y_avg = (landmarks[0][1] + landmarks[1][1]) / 2.0
            mouth_y_avg = (landmarks[3][1] + landmarks[4][1]) / 2.0
            d_em = mouth_y_avg - eye_y_avg
            if d_em < 0.05 * bh:
                continue


            # 4. Evaluación de completitud del rostro para reconocimiento biométrico.
            # YuNet SIEMPRE reporta 5 landmarks incluso cuando partes de la cara no son
            # realmente visibles (estima/hallucina posiciones). Por ello necesitamos
            # verificar no solo que los landmarks estén dentro del frame, sino que la
            # geometría facial sea coherente y los ojos tengan espacio real visible.
            edge_margin = 4  # Margen mínimo desde el borde del frame (px)

            # A. Verificar que los 5 landmarks están dentro del frame con margen
            all_landmarks_inside = True
            for lm in landmarks:
                lm_x, lm_y = float(lm[0]), float(lm[1])
                if lm_x < edge_margin or lm_x > (w - edge_margin) or lm_y < edge_margin or lm_y > (h - edge_margin):
                    all_landmarks_inside = False
                    break

            # B. Verificar que los ojos tienen espacio real por encima dentro del bbox.
            # En una cara normal, los ojos están a ~35-40% desde la parte superior.
            # Si los ojos están a < 6% del tope del bbox, la frente está severamente
            # cortada o la cabeza está inclinada hacia atrás (ojos no realmente visibles).
            # YuNet hallucina la posición de los ojos en estos casos.
            min_eye_y = min(landmarks[0][1], landmarks[1][1])
            has_visible_eyes = (min_eye_y - by) >= (0.06 * bh)

            # C. Verificar que la boca no está en el borde inferior (cara cortada por abajo)
            mouth_bottom = max(landmarks[3][1], landmarks[4][1])
            has_chin_space = ((by + bh) - mouth_bottom) >= (0.04 * bh)

            # D. Verificar estructura facial vertical intacta (ojos arriba de nariz, nariz arriba de boca)
            valid_vertical_span = d_em >= (0.15 * bh)

            # E. Verificar orden vertical correcto: ojos encima de nariz, nariz encima de boca.
            # Si la cabeza está tan inclinada que los ojos están debajo de la nariz
            # (o al mismo nivel), la geometría está colapsada.
            nose_y = landmarks[2][1]
            eyes_above_nose = min_eye_y < (nose_y - 2)
            nose_above_mouth = nose_y < (mouth_bottom - 2)

            is_complete = bool(
                all_landmarks_inside and has_visible_eyes and
                has_chin_space and valid_vertical_span and
                eyes_above_nose and nose_above_mouth
            )

            t_nose, frontal_score = compute_projective_orientation(landmarks, bw, bh)

            raw_native = raw.copy()
            raw_native[0] = bx
            raw_native[1] = by
            raw_native[2] = bw
            raw_native[3] = bh
            raw_native[4:14:2] *= scale_x
            raw_native[5:14:2] *= scale_y

            detections.append(
                FaceDetection(
                    bbox=(bx, by, bw, bh),
                    confidence=round(score, 4),
                    landmarks=landmarks,
                    area=bw * bh,
                    head_orientation_proxy=t_nose,
                    orientation_frontal_score=frontal_score,
                    raw_detection=raw_native,
                    is_complete_face=is_complete
                )
            )

        return detections, infer_ms
