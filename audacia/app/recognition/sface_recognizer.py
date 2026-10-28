"""
Módulo de Reconocimiento Facial basado en OpenCV FaceRecognizerSF (SFace).
Modelo oficial ONNX (37 MB, Licencia Apache 2.0).
Proporciona:
- Alineación facial geométrica (112x112 px)
- Extracción de vector de características de 128 dimensiones (float32)
- Cálculo de similitud coseno (Cosine Similarity: mayor valor = mayor similitud)
- Validación de calidad de imagen para registro
"""

import os
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

from audacia.app.core.paths import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class FaceQualityReport:
    is_valid: bool
    reason: str
    height: int
    confidence: float
    frontal_score: float
    sharpness: float


class SFaceRecognizer:
    """
    Wrapper de alta eficiencia sobre cv2.FaceRecognizerSF.
    Ejecuta en CPU de forma nativa sin dependencias pesadas de PyTorch o CUDA.
    """

    def __init__(self, model_path: str = "models/face_recognition_sface_2021dec.onnx"):
        self.model_path = resolve_path(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo SFace en '{self.model_path}'. "
                "Verifique que el archivo ONNX exista en el directorio de modelos."
            )

        self.recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")
        logger.info(f"SFaceRecognizer inicializado con éxito desde '{model_path}'")

    @staticmethod
    def canonical_illumination_norm(aligned_face: np.ndarray) -> np.ndarray:
        """
        Normaliza canónicamente la iluminación de un recorte facial de 112x112:
        1. Transforma al espacio perceptual de color LAB (desacopla luminancia L de color A, B).
        2. Aplica CLAHE adaptativo local al canal de luminancia.
        3. Estandariza la distribución de luminancia a un valor medio canónico (media ~128, std ~45)
           garantizando invarianza frente a contraluz, penumbra o sobreexposición.
        4. Reconstruye el rostro a BGR preservando textura, landmarks y geometría.
        """
        if aligned_face is None or aligned_face.size == 0:
            return aligned_face

        lab = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        curr_mean = float(np.mean(l_clahe))
        curr_std = float(np.std(l_clahe))

        if curr_std > 1.0:
            l_canonical = ((l_clahe - curr_mean) / curr_std) * 45.0 + 128.0
            l_canonical = np.clip(l_canonical, 0, 255).astype(np.uint8)
        else:
            l_canonical = l_clahe

        lab_canonical = cv2.merge([l_canonical, a_channel, b_channel])
        return cv2.cvtColor(lab_canonical, cv2.COLOR_LAB2BGR)

    @staticmethod
    def generate_illumination_variants(aligned_face: np.ndarray) -> list:
        """
        A partir de un recorte alineado de alta calidad, genera variantes
        fotométricas fotonaturales (sin distorsión de landmarks ni ruido CLAHE)
        para cubrir ambientes con penumbra o sobreexposición.
        Retorna lista de tuplas: [(variant_name, aligned_face_bgr), ...]
        """
        if aligned_face is None or aligned_face.size == 0:
            return []

        variants = []
        # 1. Recorte nativo original de alta fidelidad
        variants.append(("standard", aligned_face))

        # 2. Variante de iluminación baja / penumbra (gamma suave 0.75)
        inv_gamma_dark = 1.0 / 0.75
        lut_dark = np.array([((i / 255.0) ** inv_gamma_dark) * 255 for i in range(256)]).astype(np.uint8)
        dim_face = cv2.LUT(aligned_face, lut_dark)
        variants.append(("dim_augmented", dim_face))

        # 3. Variante de iluminación alta / sobreexpuesta (gamma suave 1.35)
        inv_gamma_bright = 1.0 / 1.35
        lut_bright = np.array([((i / 255.0) ** inv_gamma_bright) * 255 for i in range(256)]).astype(np.uint8)
        bright_face = cv2.LUT(aligned_face, lut_bright)
        variants.append(("bright_augmented", bright_face))

        return variants

    def align_crop(self, frame: np.ndarray, raw_detection: np.ndarray, normalize: bool = False) -> np.ndarray:
        """
        Alinea y recorta el rostro a 112x112 píxeles usando los 5 landmarks.
        raw_detection debe ser el vector nativo de 15 floats devuelto por YuNet.
        Retorna el recorte nativo limpio de 112x112 optimizado para las capas convolucionales de SFace.
        """
        if raw_detection is None or len(raw_detection) < 15:
            raise ValueError("raw_detection debe tener al menos 15 elementos (salida de YuNet).")
        raw_float = np.ascontiguousarray(raw_detection, dtype=np.float32)
        return self.recognizer.alignCrop(frame, raw_float)

    def extract_feature(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extrae el embedding facial normalizado de 128 dimensiones.
        Retorna array shape (1, 128) dtype float32.
        """
        return self.recognizer.feature(aligned_face)

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Calcula la similitud coseno entre dos embeddings de 128 dimensiones.
        Métrica: cv2.FaceRecognizerSF_FR_COSINE
        Rango: [-1.0, 1.0] donde:
          1.0  = Idéntico
          >0.6 = Mismo individuo con alta probabilidad
          <0.4 = Personas diferentes
        """
        f1 = np.ascontiguousarray(feat1, dtype=np.float32)
        f2 = np.ascontiguousarray(feat2, dtype=np.float32)
        score = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)

    @staticmethod
    def validate_registration_quality(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        frontal_score: float,
        min_height: int = 70,
        min_confidence: float = 0.65,
        min_frontal_score: float = 0.60,
        min_sharpness: float = 30.0,
        min_brightness: float = 8.0
    ) -> FaceQualityReport:
        """
        Evalúa si un rostro cumple con los estándares mínimos de nitidez óptica,
        tamaño, iluminación y orientación frontal para ser registrado con alta fidelidad.
        """
        x, y, w, h = bbox
        if h < min_height:
            return FaceQualityReport(
                is_valid=False,
                reason=f"Rostro demasiado pequeño ({h}px < {min_height}px). Acércate más a la cámara.",
                height=h, confidence=confidence, frontal_score=frontal_score, sharpness=0.0
            )

        if confidence < min_confidence:
            return FaceQualityReport(
                is_valid=False,
                reason=f"Confianza de detección insuficiente ({confidence:.2f} < {min_confidence:.2f}). Mira hacia la cámara.",
                height=h, confidence=confidence, frontal_score=frontal_score, sharpness=0.0
            )

        if frontal_score < min_frontal_score:
            return FaceQualityReport(
                is_valid=False,
                reason=f"Rostro no orientado frontalmente ({frontal_score:.2f} < {min_frontal_score:.2f}). Mira de frente.",
                height=h, confidence=confidence, frontal_score=frontal_score, sharpness=0.0
            )

        # Recortar rostro
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return FaceQualityReport(
                is_valid=False, reason="Recorte de rostro vacío.",
                height=h, confidence=confidence, frontal_score=frontal_score, sharpness=0.0
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        # Validar iluminación mínima absoluta
        if mean_brightness < min_brightness:
            return FaceQualityReport(
                is_valid=False,
                reason=f"Rostro en oscuridad extrema (brillo {mean_brightness:.1f} < {min_brightness:.1f}). Añade luz al rostro.",
                height=h, confidence=confidence, frontal_score=frontal_score, sharpness=0.0
            )

        # Medir nitidez óptica real normalizando el contraste para eliminar la dependencia cuadrática del nivel de luz
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        norm_gray = clahe.apply(gray)
        sharpness = float(cv2.Laplacian(norm_gray, cv2.CV_64F).var())

        if sharpness < min_sharpness:
            return FaceQualityReport(
                is_valid=False,
                reason=f"Imagen desenfocada o con movimiento (nitidez {sharpness:.1f} < {min_sharpness:.1f}). Mantente quieto y enfocado.",
                height=h, confidence=confidence, frontal_score=frontal_score, sharpness=round(sharpness, 1)
            )

        return FaceQualityReport(
            is_valid=True,
            reason="Calidad adecuada para registro.",
            height=h, confidence=confidence, frontal_score=frontal_score, sharpness=round(sharpness, 1)
        )
