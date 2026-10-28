"""
Servicio de Registro de Identidades (RegistrationService).
Permite registrar una persona mediante dos métodos:
1. Fotografía estática (validando exactamente 1 rostro y calidad mínima).
2. Cuadro en vivo capturado desde la cámara Logitech C270.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import cv2

from audacia.app.detection.yunet_detector import YuNetDetector
from audacia.app.recognition.sface_recognizer import SFaceRecognizer
from audacia.app.storage.identity_store import IdentityStore

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    success: bool
    message: str
    person_id: Optional[str] = None
    name: Optional[str] = None
    faces_detected: int = 0
    thumbnail: Optional[np.ndarray] = None


class RegistrationService:
    """
    Gestiona el flujo de validación y registro de nuevas personas en AudacIA.
    """

    def __init__(
        self,
        detector: YuNetDetector,
        recognizer: SFaceRecognizer,
        identity_store: IdentityStore
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.identity_store = identity_store

    def register_from_image_path(
        self,
        image_path: str,
        person_name: str,
        notes: str = ""
    ) -> RegistrationResult:
        """
        Método A: Registra una persona a partir de una fotografía en disco.
        Exige exactamente un rostro de calidad adecuada.
        """
        if not os.path.isfile(image_path):
            return RegistrationResult(
                success=False,
                message=f"El archivo de imagen no existe en '{image_path}'."
            )

        frame = cv2.imread(image_path)
        if frame is None or frame.size == 0:
            return RegistrationResult(
                success=False,
                message="No fue posible decodificar la imagen proporcionada."
            )

        return self.register_from_frame(frame, person_name, source_type="photo", notes=notes)

    def register_from_frame(
        self,
        frame: np.ndarray,
        person_name: str,
        source_type: str = "camera",
        notes: str = ""
    ) -> RegistrationResult:
        """
        Valida el cuadro y registra la persona en la base de identidades.
        """
        name_clean = person_name.strip()
        if not name_clean:
            return RegistrationResult(
                success=False,
                message="El nombre de la persona no puede estar vacío."
            )

        # 1. Detección facial con YuNet
        faces, _ = self.detector.detect(frame)
        num_faces = len(faces)

        # 2. Validación de conteo estricto
        if num_faces == 0:
            return RegistrationResult(
                success=False,
                faces_detected=0,
                message="No se encontró un rostro utilizable en la imagen."
            )

        if num_faces > 1:
            return RegistrationResult(
                success=False,
                faces_detected=num_faces,
                message=f"La imagen contiene {num_faces} rostros. Selecciona una imagen con una sola persona."
            )

        face = faces[0]

        # 3. Validación de calidad física
        q_report = self.recognizer.validate_registration_quality(
            frame=frame,
            bbox=face.bbox,
            confidence=face.confidence,
            frontal_score=face.orientation_frontal_score,
            min_height=70,
            min_confidence=0.65,
            min_frontal_score=0.60,
            min_sharpness=30.0
        )

        if not q_report.is_valid:
            return RegistrationResult(
                success=False,
                faces_detected=1,
                message=f"No puedo registrar esta imagen con suficiente calidad: {q_report.reason}"
            )

        # 4. Alineación geométrica y generación de perfil fotométrico multi-iluminación
        try:
            raw_float = np.ascontiguousarray(face.raw_detection, dtype=np.float32)
            aligned_base = self.recognizer.recognizer.alignCrop(frame, raw_float)

            # Generar variantes fotométricas canónicas (estándar, penumbra simulada, luz alta simulada)
            variants = self.recognizer.generate_illumination_variants(aligned_base)
            if not variants:
                variants = [("standard", aligned_base)]

            # Comprimir thumbnail de 112x112 en JPEG
            ret, thumb_buf = cv2.imencode(".jpg", variants[0][1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumb_bytes = thumb_buf.tobytes() if ret else None

            # 5. Persistir en SQLite (IdentityStore) con protección contra colisión de homónimos
            existing = self.identity_store.find_person_by_name(name_clean)
            if existing:
                pid = existing["person_id"]
                existing_embs = self.identity_store.get_embeddings_by_person_id(pid)

                # Comparar la nueva foto contra los rasgos biométricos históricos de esta persona
                first_feat = self.recognizer.extract_feature(variants[0][1])
                max_sim = max([float(self.recognizer.compute_similarity(first_feat, prev_emb)) for prev_emb in existing_embs]) if existing_embs else 1.0

                # Umbral de coherencia de identidad (0.32):
                # Si es menor a 0.32, son dos personas completamente distintas con el mismo nombre.
                if max_sim < 0.32:
                    logger.warning(
                        f"Rechazado intento de sobreescritura de '{name_clean}' por colisión de homónimos "
                        f"(Similitud: {max_sim:.2f} < 0.32)"
                    )
                    return RegistrationResult(
                        success=False,
                        faces_detected=1,
                        message=(
                            f"Ya existe una persona registrada como '{name_clean}', pero sus rasgos faciales "
                            f"no coinciden (Similitud: {max_sim:.2f} < 0.32). "
                            f"Si se trata de otra persona con el mismo nombre, por favor especifica un apellido "
                            f"o diferenciador (ej. '{name_clean} Gómez')."
                        )
                    )

                # Si la similitud es >= 0.32, es la misma persona actualizando su perfil (ej. barba afeitada, gafas)
                for v_name, v_crop in variants:
                    v_feat = self.recognizer.extract_feature(v_crop)
                    self.identity_store.add_embedding_to_person(
                        person_id=pid,
                        embedding=v_feat,
                        source_type=f"{source_type}_{v_name}",
                        thumbnail_bytes=thumb_bytes
                    )
                msg = f"Perfil de '{name_clean}' enriquecido exitosamente con su nuevo aspecto (Similitud: {max_sim:.2f})."
            else:
                first_name, first_crop = variants[0]
                first_feat = self.recognizer.extract_feature(first_crop)
                pid = self.identity_store.create_person(
                    name=name_clean,
                    embedding=first_feat,
                    source_type=f"{source_type}_{first_name}",
                    thumbnail_bytes=thumb_bytes,
                    notes=notes
                )
                for v_name, v_crop in variants[1:]:
                    v_feat = self.recognizer.extract_feature(v_crop)
                    self.identity_store.add_embedding_to_person(
                        person_id=pid,
                        embedding=v_feat,
                        source_type=f"{source_type}_{v_name}",
                        thumbnail_bytes=thumb_bytes
                    )
                msg = f"'{name_clean}' registrado exitosamente con perfil biométrico invariante a la iluminación."

            logger.info(f"Registro exitoso: {msg} (ID: {pid})")
            return RegistrationResult(
                success=True,
                message=msg,
                person_id=pid,
                name=name_clean,
                faces_detected=1,
                thumbnail=variants[0][1]
            )

        except Exception as e:
            logger.error(f"Falla durante el proceso de alineación o guardado: {e}")
            return RegistrationResult(
                success=False,
                faces_detected=1,
                message=f"Error interno procesando el rostro: {e}"
            )
