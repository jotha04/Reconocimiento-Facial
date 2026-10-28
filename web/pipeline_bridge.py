"""
FR2000 - Puente de Procesamiento y Streaming en Tiempo Real
Conecta el pipeline biométrico de visión, atención y reconocimiento facial
con el servidor web asíncrono para telemetría y video MJPEG.
"""

import os
import sys
import time
import threading
import logging
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

# Asegurar importación de módulos raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audacia.app.camera.device_discovery import find_c270_camera
from audacia.app.camera.camera_stream import CameraStream, MockCameraStream
from audacia.app.detection.yunet_detector import YuNetDetector
from audacia.app.tracking.face_tracker import FaceTracker
from audacia.app.tracking.track import Track, TrackState
from audacia.app.attention.attention_engine import AttentionEngine
from audacia.app.recognition.sface_recognizer import SFaceRecognizer
from audacia.app.storage.identity_store import IdentityStore
from audacia.app.recognition.recognition_engine import RecognitionEngine
from audacia.app.recognition.registration_service import RegistrationService
from audacia.app.interaction.interaction_engine import InteractionEngine
from audacia.app.attention.config import load_audacia_config
from audacia.app.diagnostics.metrics_tracker import MetricsTracker
from audacia.app.diagnostics.visualizer import DiagnosticVisualizer

logger = logging.getLogger("FR2000-Pipeline")


class PipelineBridge:
    """
    Gestor maestro del ciclo de procesamiento biométrico FR2000.
    Ejecuta el pipeline completo de visión, tracking, atención y reconocimiento en hilo dedicado.
    """

    def __init__(self, config_path: str = "config/audacia_config.yaml", use_mock: bool = False):
        self.config_path = config_path
        self.config_dict = load_audacia_config(config_path)

        self.camera_cfg = self.config_dict["camera"]
        self.detection_cfg = self.config_dict.get("detection")
        self.tracking_cfg = self.config_dict["tracking"]
        self.attention_cfg = self.config_dict["attention"]
        self.recognition_cfg = self.config_dict["recognition"]
        self.interaction_cfg = self.config_dict["interaction"]
        self.greetings_cfg = self.config_dict["greetings"]
        self.audio_cfg = self.config_dict["audio"]

        self.use_mock = use_mock
        self.is_running = False
        self.is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Modos de visualización biométrica: 'biometric' | 'focus' | 'clean'
        self.view_mode = "focus"

        # Buffer del último frame codificado en JPEG y sincronización de eventos
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_raw: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._new_frame_event = threading.Event()

        # Historial de telemetría para el gráfico en tiempo real (últimos 60 puntos, ~30s)
        self.telemetry_history: deque = deque(maxlen=60)
        self.recent_events: deque = deque(maxlen=20)

        # Telemetría instantánea
        self.current_telemetry: Dict[str, Any] = {
            "fps_visual": 30.0,
            "fps_detection": 28.5,
            "latency_ms": 42.0,
            "attention_score": 0.0,
            "attention_level": "ESPERA",
            "active_tracks_count": 0,
            "known_persons_count": 0,
            "current_session_id": None,
            "current_session_state": "NO_INTERACTION",
            "last_greeting_text": "Sistema inicializado",
            "last_greeting_time": "--:--:--",
            "last_greeting_latency": "0.2 ms (Caché)",
            "device_name": "Logitech C920 Full HD",
            "camera_status": "CONECTADA",
            "score_threshold": self.detection_cfg.score_threshold if self.detection_cfg else 0.65,
            "similarity_threshold": self.recognition_cfg.similarity_threshold,
            "primary_subject": {
                "name": "En Espera",
                "similarity": 0.0,
                "decision": "EN ESPERA",
                "state": "ESPERA",
                "cooldown": "Disponible",
                "recognized_persons": []
            },
            "registered_identities": [],
            "participants": []
        }

        # Inicialización de componentes
        self._init_pipeline()

    def _init_pipeline(self):
        """Inicializa los módulos de hardware, IA y motores de decisión."""
        # 1. Cámara
        c_info = None
        if not self.use_mock:
            try:
                c_info = find_c270_camera()
                logger.info(f"[CÁMARA] Conectada: {c_info.name} (Índice {c_info.index})")
                self.current_telemetry["device_name"] = c_info.name
            except Exception as e:
                logger.warning(f"[CÁMARA] No se detectó cámara física ({e}). Usando Mock.")
                self.use_mock = True
                self.current_telemetry["device_name"] = "MockCamera (Sintética)"

        if self.use_mock:
            self.camera = MockCameraStream(
                target_width=self.camera_cfg.target_width,
                target_height=self.camera_cfg.target_height,
                target_fps=self.camera_cfg.target_fps
            )
        else:
            self.camera = CameraStream(
                device_info=c_info,
                target_width=self.camera_cfg.target_width,
                target_height=self.camera_cfg.target_height,
                target_fps=self.camera_cfg.target_fps,
                use_threaded_reader=self.camera_cfg.use_threaded_reader
            )

        # 2. Detector YuNet
        self.detector = YuNetDetector(
            model_path=self.detection_cfg.model_path if self.detection_cfg else "models/face_detection_yunet_2023mar.onnx",
            input_size=(self.camera_cfg.target_width, self.camera_cfg.target_height),
            score_threshold=self.detection_cfg.score_threshold if self.detection_cfg else 0.65,
            nms_threshold=self.detection_cfg.nms_threshold if self.detection_cfg else 0.30,
            min_face_size=self.detection_cfg.min_face_size if self.detection_cfg else 55,
            max_inference_dimension=self.camera_cfg.max_inference_dimension
        )

        # 3. Tracker cinemático
        self.tracker = FaceTracker(
            config=self.tracking_cfg,
            frame_width=self.camera_cfg.target_width,
            frame_height=self.camera_cfg.target_height
        )

        # 4. Motor de Atención
        self.attention_engine = AttentionEngine(
            config=self.attention_cfg,
            frame_width=self.camera_cfg.target_width,
            frame_height=self.camera_cfg.target_height
        )

        # 5. Reconocimiento e Identidades SQLite
        self.identity_store = IdentityStore(db_path=self.recognition_cfg.db_path)
        self.sface_recognizer = SFaceRecognizer(model_path=self.recognition_cfg.model_path)
        self.recognition_engine = RecognitionEngine(
            recognizer=self.sface_recognizer,
            identity_store=self.identity_store,
            config=self.recognition_cfg
        )
        self.registration_detector = YuNetDetector(
            model_path=self.detection_cfg.model_path if self.detection_cfg else "models/face_detection_yunet_2023mar.onnx",
            input_size=(640, 640),
            score_threshold=0.60,
            nms_threshold=0.30,
            min_face_size=50,
            max_inference_dimension=800
        )
        self.registration_service = RegistrationService(
            detector=self.registration_detector,
            recognizer=self.sface_recognizer,
            identity_store=self.identity_store
        )

        # 6. Motor de Interacción y Saludos
        self.interaction_engine = InteractionEngine(
            interaction_config=self.interaction_cfg,
            greetings_config=self.greetings_cfg,
            audio_config=self.audio_cfg
        )

        # 7. Diagnósticos y Visualizador
        self.metrics = MetricsTracker(window_size=60)
        self.visualizer = DiagnosticVisualizer()

        # 8. Warmup de modelos ONNX (elimina lag de JIT cold-start en los primeros ~15s)
        self._warmup_models()

        self._refresh_registered_identities()
        self._add_event("SISTEMA", "FR2000 Pipeline biométrico inicializado y en línea")

    def _warmup_models(self):
        """
        Ejecuta inferencias dummy sobre YuNet y SFace para forzar la compilación JIT
        del backend DNN de OpenCV antes de procesar frames reales.
        Esto elimina el lag de ~3 FPS que se observa en los primeros 15 segundos.
        """
        logger.info("[WARMUP] Precalentando modelos ONNX (YuNet + SFace)...")
        t0 = time.perf_counter()

        try:
            # Warmup YuNet detector: usar la resolución EXACTA de captura (ej. 1920x1080)
            # para compilar el grafo DNN y pre-alocar tensores de feature maps antes del streaming en vivo
            target_w = self.camera_cfg.target_width
            target_h = self.camera_cfg.target_height
            dummy_frame = np.random.randint(40, 200, (target_h, target_w, 3), dtype=np.uint8)

            # Dibujar un "rostro" sintético para activar las capas NMS también
            cx, cy = target_w // 2, target_h // 2
            cv2.ellipse(dummy_frame, (cx, cy), (120, 160), 0, 0, 360, (180, 160, 140), -1)
            cv2.circle(dummy_frame, (cx - 40, cy - 30), 12, (40, 40, 40), -1)  # Ojo izq
            cv2.circle(dummy_frame, (cx + 40, cy - 30), 12, (40, 40, 40), -1)  # Ojo der
            cv2.ellipse(dummy_frame, (cx, cy + 50), (30, 16), 0, 0, 360, (120, 80, 80), -1)  # Boca

            # Ejecutar 3 inferencias para estabilizar el JIT
            for i in range(3):
                self.detector.detect(dummy_frame)

            warmup_det_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(f"[WARMUP] YuNet precalentado a {target_w}x{target_h} en {warmup_det_ms:.0f} ms (3 inferencias)")

            # Warmup SFace recognizer: crear un recorte dummy de 112x112 y extraer features
            t1 = time.perf_counter()
            dummy_aligned = np.random.randint(40, 200, (112, 112, 3), dtype=np.uint8)
            for i in range(3):
                self.sface_recognizer.extract_feature(dummy_aligned)

            warmup_rec_ms = (time.perf_counter() - t1) * 1000.0
            logger.info(f"[WARMUP] SFace precalentado en {warmup_rec_ms:.0f} ms (3 inferencias)")

            # Warmup del registration_detector también (usa input_size diferente)
            t2 = time.perf_counter()
            dummy_reg = np.random.randint(40, 200, (480, 640, 3), dtype=np.uint8)
            for i in range(2):
                self.registration_detector.detect(dummy_reg)

            warmup_reg_ms = (time.perf_counter() - t2) * 1000.0

            total_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(f"[WARMUP] Modelos ONNX precalentados exitosamente en {total_ms:.0f} ms total")

        except Exception as e:
            logger.warning(f"[WARMUP] Error durante precalentamiento (no crítico): {e}")

    def _refresh_registered_identities(self):
        """Actualiza la lista de personas registradas para el panel web."""
        persons = self.identity_store.list_persons()
        self.current_telemetry["registered_identities"] = [
            {"id": p["person_id"], "name": p["name"], "samples": p["sample_count"]}
            for p in persons
        ]

    def _add_event(self, category: str, text: str):
        """Agrega un evento con timestamp al registro en vivo."""
        timestamp = time.strftime("%H:%M:%S")
        self.recent_events.appendleft({
            "timestamp": timestamp,
            "category": category,
            "text": text
        })

    def start(self):
        """Inicia el hilo de ejecución del pipeline."""
        if self.is_running:
            return
        self.is_running = True
        self.camera.open()
        time.sleep(0.3)
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="FR2000-Worker")
        self._thread.start()
        logger.info("[FR2000] Hilo de procesamiento biométrico iniciado.")

    def stop(self):
        """Detiene el pipeline y libera los dispositivos."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.camera.close()
        self.interaction_engine.audio_queue.stop()
        logger.info("[FR2000] Pipeline detenido.")

    def toggle_pause(self) -> bool:
        """Pausa o reanuda la captura de video."""
        self.is_paused = not self.is_paused
        status = "PAUSADO" if self.is_paused else "EN VIVO"
        self._add_event("STREAM", f"Captura {status}")
        return self.is_paused

    def set_score_threshold(self, threshold: float):
        """Ajusta dinámicamente el umbral de detección de YuNet."""
        th = float(np.clip(threshold, 0.30, 0.95))
        self.detector.score_threshold = th
        self.current_telemetry["score_threshold"] = round(th, 2)
        self._add_event("CONTROL", f"Umbral YuNet calibrado a {th:.2f}")

    def set_similarity_threshold(self, threshold: float):
        """Ajusta dinámicamente el umbral de similitud SFace."""
        th = float(np.clip(threshold, 0.35, 0.85))
        self.recognition_cfg.similarity_threshold = th
        self.recognition_engine.config.similarity_threshold = th
        self.current_telemetry["similarity_threshold"] = round(th, 2)
        self._add_event("CONTROL", f"Umbral Similitud SFace calibrado a {th:.2f}")

    def set_audio_enabled(self, enabled: bool):
        """Activa o desactiva la síntesis de saludos por voz."""
        self.audio_cfg.enabled = enabled
        if hasattr(self.interaction_engine, "audio_queue") and self.interaction_engine.audio_queue:
            self.interaction_engine.audio_queue.config.enabled = enabled
        status_txt = "ACTIVADOS" if enabled else "SILENCIADOS"
        self._add_event("AUDIO", f"Saludos por voz {status_txt}")

    def set_view_mode(self, mode: str):
        """Establece el modo de visualización: biometric, focus, clean."""
        if mode in ("biometric", "focus", "clean"):
            self.view_mode = mode
            self._add_event("VISTA", f"Modo de video cambiado a '{mode}'")

    def reload_identities(self) -> int:
        """Recarga el catálogo biométrico desde SQLite."""
        self.recognition_engine.reload_identities()
        self._refresh_registered_identities()
        count = len(self.recognition_engine._cached_embeddings)
        self._add_event("BIOMETRÍA", f"Catálogo recargado: {count} firmas biométricas en memoria")
        return count

    def register_from_camera(self, person_name: str) -> Dict[str, Any]:
        """
        Registra una persona tomando el fotograma actual de la cámara en vivo.
        Aplica validación de calidad y normalización multi-iluminación (CLAHE + variantes fotométricas).
        """
        if self._latest_frame_raw is None:
            return {"success": False, "message": "No hay fotograma de cámara disponible aún."}

        frame = self._latest_frame_raw.copy()
        res = self.registration_service.register_from_frame(frame, person_name, source_type="camera")
        if res.success:
            self.reload_identities()
            self._add_event("REGISTRO", f"Persona '{person_name}' registrada desde cámara con normalización.")
        else:
            self._add_event("ERROR", f"Fallo al registrar '{person_name}': {res.message}")
        return {"success": res.success, "message": res.message, "person_id": res.person_id}

    def register_from_image_bytes(self, image_bytes: bytes, person_name: str) -> Dict[str, Any]:
        """
        Registra una persona a partir de una fotografía subida (JPG/PNG).
        Aplica validación de calidad y normalización multi-iluminación (CLAHE + variantes fotométricas).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return {"success": False, "message": "No se pudo decodificar el archivo de imagen subido."}

        res = self.registration_service.register_from_frame(frame, person_name, source_type="photo")
        if res.success:
            self.reload_identities()
            self._add_event("REGISTRO", f"Persona '{person_name}' registrada desde foto con normalización.")
        else:
            self._add_event("ERROR", f"Fallo al registrar '{person_name}': {res.message}")
        return {"success": res.success, "message": res.message, "person_id": res.person_id}

    def delete_person(self, person_id: str) -> Dict[str, Any]:
        """Elimina una persona y sus firmas biométricas de la base de datos."""
        success = self.identity_store.delete_person(person_id)
        if success:
            self.reload_identities()
            self._add_event("REGISTRO", f"Identidad eliminada de la base de datos.")
            return {"success": True, "message": "Identidad eliminada correctamente."}
        return {"success": False, "message": "No se encontró la identidad solicitada."}

    def save_snapshot(self) -> Optional[str]:
        """Guarda un fotograma actual en disco como evidencia diagnóstica."""
        if self._latest_frame_raw is None:
            return None
        snap_dir = "data/snapshots"
        os.makedirs(snap_dir, exist_ok=True)
        filename = f"fr2000_snap_{int(time.time())}.jpg"
        filepath = os.path.join(snap_dir, filename)
        cv2.imwrite(filepath, self._latest_frame_raw)
        self._add_event("SNAPSHOT", f"Captura biométrica guardada: {filename}")
        return filepath

    def _render_biometric_view(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Vista Biometría Completa: Bounding box, landmarks faciales y badge de identidad.
        Analiza toda la sección de la cámara sin líneas divisorias.
        """
        vis = self.visualizer.render(
            frame=frame,
            tracks=tracks,
            metrics_summary={},
            camera_telemetry={},
            interaction_telemetry=self.interaction_engine.get_telemetry()
        )
        if vis.shape[1] > 1280:
            vis = cv2.resize(vis, (1280, 720), interpolation=cv2.INTER_AREA)
        return vis

    def _render_focus_view(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Vista Enfoque Facial: Minimalista y elegante, con cajas perimetrales y nombres sin sobrecarga.
        """
        vis = frame.copy()
        for t in tracks:
            if t.state == TrackState.DISENGAGED:
                continue
            x, y, w, h = t.bbox

            # Rostros incompletos (cortados por el borde): mostrar indicador visual sutil
            if not getattr(t, 'is_complete_face', True):
                color = (120, 120, 120)  # Gris: rostro parcial, no se evalúa
                label = ""  # Sin etiqueta para no confundir
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
                continue

            if t.identity_decision == "KNOWN":
                color = (50, 255, 120)  # Verde
                label = t.identity_label
            elif t.identity_decision == "UNKNOWN":
                color = (40, 205, 245)  # Ámbar
                label = t.identity_label if t.identity_label.startswith("Visitante #") else (f"Visitante #{t.visitor_id}" if getattr(t, 'visitor_id', None) else f"Visitante #{t.track_id}")
            else:
                color = (40, 205, 245)  # Ámbar
                label = "Identificando..."

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            cv2.putText(vis, label, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if vis.shape[1] > 1280:
            vis = cv2.resize(vis, (1280, 720), interpolation=cv2.INTER_AREA)
        return vis

    def _process_frame(self, frame: np.ndarray, t_cycle_start: float, capture_ms: float, last_history_push: float) -> float:
        """Ejecuta las etapas de detección, tracking, atención y reconocimiento para un fotograma."""
        # B. Detección YuNet
        faces, detector_ms = self.detector.detect(frame)

        # C. Tracking cinemático
        t0 = time.perf_counter()
        tracks = self.tracker.update(faces)
        tracking_ms = (time.perf_counter() - t0) * 1000.0

        # D. Motor de Atención
        t0 = time.perf_counter()
        if self.view_mode != "clean":
            tracks = self.attention_engine.update(tracks)
        attention_ms = (time.perf_counter() - t0) * 1000.0

        # E. Reconocimiento Facial SFace
        t0 = time.perf_counter()
        if self.view_mode != "clean":
            tracks = self.recognition_engine.process_tracks(frame, tracks)
        recognition_ms = (time.perf_counter() - t0) * 1000.0

        # F. Motor de Interacción y Saludos
        session = None
        if self.view_mode != "clean":
            session = self.interaction_engine.update(tracks)

        # G. Renderizado según modo seleccionado
        t0 = time.perf_counter()
        if self.view_mode == "biometric":
            vis_frame = self._render_biometric_view(frame, tracks)
        elif self.view_mode == "focus":
            vis_frame = self._render_focus_view(frame, tracks)
        else:
            vis_frame = frame.copy()
            if vis_frame.shape[1] > 1280:
                vis_frame = cv2.resize(vis_frame, (1280, 720), interpolation=cv2.INTER_AREA)

        render_ms = (time.perf_counter() - t0) * 1000.0
        pipeline_ms = (time.perf_counter() - t_cycle_start) * 1000.0

        # Registrar latencias
        self.metrics.record_metrics(
            capture_ms=capture_ms,
            detector_ms=detector_ms,
            tracking_ms=tracking_ms,
            attention_ms=attention_ms,
            render_ms=render_ms,
            pipeline_ms=pipeline_ms,
            recognition_ms=recognition_ms
        )

        # H. Codificación JPEG para streaming web
        ret, jpeg_buf = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with self._lock:
                self._latest_jpeg = jpeg_buf.tobytes()
                self._frame_id += 1
                self._new_frame_event.set()

        # I. Extracción de telemetría y métricas
        m_summary = self.metrics.get_summary()
        fps_visual = round(m_summary.get("fps_effective", 30.0), 1)
        fps_inf = round(1000.0 / max(1.0, detector_ms), 1)

        # Identificar sujetos activos en escena (confirmados por tracker y vistos en últimos frames)
        # Excluir rostros parciales/cortados que no han sido evaluados (evita mostrar "Identificando..." para caras incompletas)
        active_tracks = [
            t for t in tracks
            if t.consecutive_misses <= 3 and (
                getattr(t, 'is_complete_face', True) or
                t.identity_decision in ("KNOWN", "UNKNOWN")
            )
        ]

        participants_list = []
        known_active = []
        seen_known_names = set()

        for t in active_tracks:
            # Identificar si es conocido (directo o por historial consolidado reciente)
            is_known = (t.identity_decision == "KNOWN") or (
                getattr(t, 'recognition_candidate_history', None) and len(t.recognition_candidate_history) > 0 and t.recognition_candidate_history[-1][0] == "KNOWN"
            )
            resolved_name = t.identity_label if t.identity_decision == "KNOWN" else (
                t.recognition_candidate_history[-1][1] if is_known else t.identity_label
            )
            person_id = getattr(t, "person_id", None)

            # Cálculo de cooldown individual con sincronización de sesión activa
            is_in_cd = False
            cd_rem = 0.0
            cd_str = "Disponible"

            if is_known:
                if self.interaction_engine.cooldown_manager.is_person_in_cooldown(person_id, resolved_name):
                    cd_rem = self.interaction_engine.cooldown_manager.get_remaining_cooldown(person_id, resolved_name)
                    if cd_rem > 0:
                        is_in_cd = True
                        cd_str = f"En Cooldown ({int(cd_rem)}s)"
                elif session and t.track_id in session.participants:
                    sp = session.participants[t.track_id]
                    if sp.greeted_in_session:
                        elapsed = time.monotonic() - (sp.greeted_at or time.monotonic())
                        cd_rem = max(1.0, self.interaction_engine.cooldown_manager.person_cooldown_seconds - elapsed)
                        is_in_cd = True
                        cd_str = f"En Cooldown ({int(cd_rem)}s)"
                    elif session.active_batch and any(p.track_id == t.track_id for p in session.active_batch.participants):
                        cd_rem = self.interaction_engine.cooldown_manager.person_cooldown_seconds
                        is_in_cd = True
                        cd_str = "En Cooldown"
            elif t.identity_decision == "UNKNOWN":
                c_x = t.bbox[0] + t.bbox[2] / 2.0
                c_y = t.bbox[1] + t.bbox[3] / 2.0
                cd_rem = self.interaction_engine.cooldown_manager.get_unknown_remaining_cooldown(
                    track_id=t.track_id, visitor_id=getattr(t, "visitor_id", None), center_x=c_x, center_y=c_y, bbox_height=t.bbox[3], embedding=getattr(t, "embedding", None)
                )
                if cd_rem > 0:
                    is_in_cd = True
                    cd_str = f"En Cooldown ({int(cd_rem)}s)"
                elif session and t.track_id in session.participants and session.participants[t.track_id].greeted_in_session:
                    is_in_cd = True
                    cd_str = "En Cooldown"

            v_label = t.identity_label if t.identity_label.startswith("Visitante #") else (f"Visitante #{t.visitor_id}" if getattr(t, 'visitor_id', None) else f"Visitante #{t.track_id}")
            p_item = {
                "track_id": t.track_id,
                "label": resolved_name if is_known else (v_label if t.identity_decision == "UNKNOWN" else "Identificando..."),
                "decision": "KNOWN" if is_known else t.identity_decision,
                "similarity": round(t.recognition_similarity * 100, 1),
                "attention_score": round(t.attention_score * 100, 1),
                "state": t.state.name,
                "in_cooldown": is_in_cd,
                "cooldown_remaining": round(cd_rem, 1),
                "cooldown": cd_str
            }
            participants_list.append(p_item)

            if is_known and resolved_name:
                clean_name = resolved_name.strip()
                if clean_name not in seen_known_names:
                    seen_known_names.add(clean_name)
                    known_active.append({
                        "track_id": t.track_id,
                        "name": clean_name,
                        "person_id": person_id,
                        "similarity": round(t.recognition_similarity * 100, 1),
                        "attention_score": round(t.attention_score * 100, 1),
                        "state": t.state.name,
                        "in_cooldown": is_in_cd,
                        "cooldown_remaining": round(cd_rem, 1),
                        "cooldown": cd_str
                    })

        # Configurar primary_subject_data considerando múltiples personas detectadas
        if len(known_active) > 1:
            names_str = ", ".join(k["name"] for k in known_active)
            avg_sim = round(sum(k["similarity"] for k in known_active) / len(known_active), 1)
            cd_active = [k for k in known_active if k["in_cooldown"]]
            if len(cd_active) == len(known_active):
                max_cd = max(k["cooldown_remaining"] for k in known_active)
                group_cd_str = f"En Cooldown ({int(max_cd)}s)"
            elif len(cd_active) > 0:
                group_cd_str = f"Parcial ({len(cd_active)}/{len(known_active)})"
            else:
                group_cd_str = "Disponible"

            group_state = "ENGAGED" if any(t.state == TrackState.ENGAGING for t in active_tracks) else (
                "CANDIDATE" if any(t.state == TrackState.ATTENTION_CANDIDATE for t in active_tracks) else (
                    "OBSERVED" if any(t.state == TrackState.OBSERVED for t in active_tracks) else "DESENGANCHADO"
                )
            )
            primary_subject_data = {
                "name": names_str,
                "similarity": avg_sim,
                "decision": "KNOWN_GROUP",
                "state": group_state,
                "cooldown": group_cd_str,
                "recognized_persons": known_active
            }
        elif len(known_active) == 1:
            k = known_active[0]
            if len(participants_list) > 1:
                all_chips = [
                    {
                        "track_id": p["track_id"],
                        "name": p["label"],
                        "similarity": p["similarity"],
                        "attention_score": p["attention_score"],
                        "state": p["state"],
                        "in_cooldown": p["in_cooldown"],
                        "cooldown_remaining": p["cooldown_remaining"],
                        "cooldown": p["cooldown"]
                    }
                    for p in participants_list
                ]
                primary_subject_data = {
                    "name": f"{k['name']} (+{len(participants_list) - 1} en escena)",
                    "similarity": k["similarity"],
                    "decision": "KNOWN",
                    "state": k["state"],
                    "cooldown": k["cooldown"],
                    "recognized_persons": all_chips
                }
            else:
                primary_subject_data = {
                    "name": k["name"],
                    "similarity": k["similarity"],
                    "decision": "KNOWN",
                    "state": k["state"],
                    "cooldown": k["cooldown"],
                    "recognized_persons": known_active
                }
        elif len(active_tracks) > 0:
            if len(participants_list) > 1:
                all_chips = [
                    {
                        "track_id": p["track_id"],
                        "name": p["label"],
                        "similarity": p["similarity"],
                        "attention_score": p["attention_score"],
                        "state": p["state"],
                        "in_cooldown": p["in_cooldown"],
                        "cooldown_remaining": p["cooldown_remaining"],
                        "cooldown": p["cooldown"]
                    }
                    for p in participants_list
                ]
                primary_subject_data = {
                    "name": f"{len(participants_list)} Personas en Escena",
                    "similarity": 0.0,
                    "decision": "MULTIPLE_UNKNOWN",
                    "state": "ENGAGED" if any(t.state == TrackState.ENGAGING for t in active_tracks) else "OBSERVED",
                    "cooldown": "Disponible",
                    "recognized_persons": all_chips
                }
            else:
                best_t = max(active_tracks, key=lambda tr: tr.attention_score)
                match_p = participants_list[0] if participants_list else None
                u_cd = match_p["cooldown"] if match_p else "Disponible"
                best_v_label = best_t.identity_label if best_t.identity_label.startswith("Visitante #") else (f"Visitante #{best_t.visitor_id}" if getattr(best_t, 'visitor_id', None) else f"Visitante #{best_t.track_id}")
                subj_name = match_p["label"] if match_p else (best_v_label if best_t.identity_decision == "UNKNOWN" else "Identificando...")
                primary_subject_data = {
                    "name": subj_name,
                    "similarity": round(best_t.recognition_similarity * 100, 1),
                    "decision": best_t.identity_decision,
                    "state": best_t.state.name,
                    "cooldown": u_cd,
                    "recognized_persons": []
                }
        else:
            latest_cd_rem, latest_who = self.interaction_engine.cooldown_manager.get_latest_greeting_remaining_cooldown()
            last_cd_str = f"En Cooldown ({int(latest_cd_rem)}s)" if latest_cd_rem > 0 else "Disponible"
            primary_subject_data = {
                "name": "En Espera",
                "similarity": 0.0,
                "decision": "EN ESPERA",
                "state": "ESPERA",
                "cooldown": last_cd_str,
                "recognized_persons": []
            }

        max_attention = max([t.attention_score for t in active_tracks], default=0.0)
        attention_pct = round(max_attention * 100.0, 1)

        # Detectar nuevos saludos emitidos para el log de eventos
        i_telemetry = self.interaction_engine.get_telemetry()
        last_text = i_telemetry.get("last_greeting_text")
        if last_text and last_text != self.current_telemetry.get("last_greeting_text") and last_text != "None":
            self._add_event("SALUDO", f"Emitido: '{last_text}'")

        # Actualizar telemetría actual
        self.current_telemetry.update({
            "fps_visual": fps_visual,
            "fps_detection": fps_inf,
            "latency_ms": round(pipeline_ms, 1),
            "attention_score": attention_pct,
            "attention_level": primary_subject_data["state"],
            "active_tracks_count": len(active_tracks),
            "known_persons_count": len([t for t in active_tracks if t.identity_decision == "KNOWN"]),
            "current_session_id": session.session_id if session else None,
            "current_session_state": session.state.name if session else "NO_INTERACTION",
            "last_greeting_text": last_text if last_text and last_text != "None" else self.current_telemetry["last_greeting_text"],
            "primary_subject": primary_subject_data,
            "participants": participants_list
        })

        # Añadir punto histórico al gráfico cada ~350 ms
        now_mono = time.monotonic()
        if now_mono - last_history_push >= 0.35:
            last_history_push = now_mono
            self.telemetry_history.append({
                "time": time.strftime("%H:%M:%S"),
                "attention": attention_pct,
                "latency": round(pipeline_ms, 1)
            })

        return last_history_push

    def _worker_loop(self):
        """Bucle principal de procesamiento a 30 FPS en hilo dedicado."""
        last_history_push = 0.0

        while self.is_running:
            t_cycle_start = self.metrics.record_frame_start()

            if self.is_paused:
                time.sleep(0.033)
                continue

            # A. Captura de frame
            ret, frame, capture_ms = self.camera.read(timeout=0.5)

            if not ret or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue

            self._latest_frame_raw = frame

            try:
                last_history_push = self._process_frame(frame, t_cycle_start, capture_ms, last_history_push)
            except Exception as e:
                logger.error(f"[FR2000-Worker] Excepción recuperada en ciclo de procesamiento: {e}", exc_info=True)

            # Control de frecuencia de bucle (~30 FPS sostenido)
            elapsed = time.perf_counter() - t_cycle_start
            sleep_time = max(0.001, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)

    def get_jpeg_frame(self) -> Optional[bytes]:
        """Retorna el último frame codificado en JPEG de manera segura."""
        with self._lock:
            return self._latest_jpeg

    def wait_for_new_frame(self, last_seen_id: int, timeout: float = 0.05) -> Tuple[int, Optional[bytes]]:
        """
        Espera hasta que un fotograma nuevo esté disponible, evitando duplicados en MJPEG.
        Retorna (nuevo_frame_id, frame_bytes).
        """
        if self._frame_id > last_seen_id:
            with self._lock:
                return self._frame_id, self._latest_jpeg

        self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        with self._lock:
            if self._frame_id > last_seen_id:
                return self._frame_id, self._latest_jpeg
            return self._frame_id, None

    def get_telemetry_payload(self) -> Dict[str, Any]:
        """Retorna la telemetría actual y el historial para el panel web."""
        return {
            "telemetry": self.current_telemetry,
            "history": list(self.telemetry_history),
            "events": list(self.recent_events)
        }
