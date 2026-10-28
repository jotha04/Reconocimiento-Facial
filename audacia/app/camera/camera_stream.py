"""
Módulo de captura de video de alta eficiencia con soporte para adquisición
en hilo secundario (non-blocking) o síncrono, y telemetría de rendimiento.
"""

import time
import threading
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

from audacia.app.camera.device_discovery import CameraDeviceInfo, find_c270_camera

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Gestiona la conexión y lectura de frames de la Logitech C270 HD WebCam.
    """

    def __init__(
        self,
        device_info: Optional[CameraDeviceInfo] = None,
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: int = 30,
        use_threaded_reader: bool = True
    ):
        self.device_info = device_info or find_c270_camera()
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.use_threaded_reader = use_threaded_reader

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.camera_healthy = False
        self._reconnect_backoff_s = 1.0

        # Telemetría interna
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_ready_event = threading.Event()

        self.frames_captured = 0
        self.frames_delivered = 0
        self.dropped_frames = 0

        self.actual_width = 0
        self.actual_height = 0
        self.reported_fps = 0.0
        self.fourcc_str = ""

        # Métricas de tiempo
        self._recent_grab_times = []
        self._recent_intervals = []
        self._last_grab_time = 0.0

    def open(self) -> bool:
        """Abre y configura el stream de la cámara (con fallback a cámaras secundarias y modo sintético)."""
        logger.info(
            f"Abriendo cámara '{self.device_info.name}' (Índice {self.device_info.index}, "
            f"Backend {self.device_info.backend}) a {self.target_width}x{self.target_height}..."
        )

        self.cap = cv2.VideoCapture(self.device_info.index, self.device_info.backend)
        if not self.cap.isOpened():
            logger.warning(f"Fallo al abrir backend {self.device_info.backend} en índice {self.device_info.index}. Intentando CAP_ANY...")
            self.cap = cv2.VideoCapture(self.device_info.index, cv2.CAP_ANY)

        if not self.cap.isOpened() and self.device_info.index != 0:
            logger.warning("Intentando abrir cámara en índice por defecto 0...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)

        if not self.cap.isOpened():
            logger.warning(
                f"[AVISO CÁMARA SINTÉTICA] No fue posible abrir el dispositivo de video físico en el índice {self.device_info.index}. "
                "Activando modo de stream sintético con diagnóstico visual..."
            )
            self.is_synthetic = True
            self.actual_width = self.target_width
            self.actual_height = self.target_height
            self.reported_fps = 30.0
            self.fourcc_str = "SYNT"
            self.camera_healthy = True

            if self.use_threaded_reader:
                self._is_running = True
                self._thread = threading.Thread(target=self._capture_worker, daemon=True)
                self._thread.start()
                self._frame_ready_event.wait(timeout=2.0)

            return True

        self.is_synthetic = False

        # Configurar compresión de hardware MJPG para evitar saturar el bus USB 2.0 (124 MB/s YUYV -> 4 MB/s MJPG)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Leer propiedades reales negociadas por el driver
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.target_width
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.target_height
        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]) if fourcc > 0 else "MJPG"

        logger.info(
            f"Cámara inicializada: {self.actual_width}x{self.actual_height} "
            f"@ {self.reported_fps:.1f} FPS (FOURCC: '{self.fourcc_str}')"
        )

        # Descartar primeros 15 frames de calentamiento para permitir convergencia de auto-exposición y ganancia
        for _ in range(15):
            self.cap.read()

        self.camera_healthy = True

        if self.use_threaded_reader:
            self._is_running = True
            self._thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._thread.start()
            # Esperar a que el primer frame esté listo
            self._frame_ready_event.wait(timeout=2.0)

        return True

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Genera un cuadro BGR de 640x480 con información visual de estado para diagnóstico."""
        frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        for y in range(self.target_height):
            v = int(20 + 25 * (y / self.target_height))
            frame[y, :] = (v, v // 2, 35)

        for x in range(0, self.target_width, 40):
            cv2.line(frame, (x, 0), (x, self.target_height), (45, 45, 55), 1)
        for y in range(0, self.target_height, 40):
            cv2.line(frame, (0, y), (self.target_width, y), (45, 45, 55), 1)

        t_str = time.strftime("%H:%M:%S")
        cv2.putText(frame, "AudacIA - STREAM DIAGNOSTICO (SIN CAMARA FISICA DETECTADA)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, f"Hora: {t_str} | Dispositivo: {self.device_info.name}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "Para activar video real:", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.putText(frame, "1. Revisa Permisos en macOS: Ajustes -> Privacidad -> Camara -> Terminal", (25, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "2. Si usas Logitech C270, reconecta el cable USB al Mac", (25, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _capture_worker(self):
        """Hilo de fondo que vacía continuamente el buffer de la cámara y maneja reconexión."""
        while self._is_running:
            if getattr(self, "is_synthetic", False):
                t_start = time.perf_counter()
                frame = self._generate_synthetic_frame()
                time.sleep(1.0 / max(1.0, self.target_fps))
                t_end = time.perf_counter()

                with self._lock:
                    self.frames_captured += 1
                    self._latest_frame = frame
                    self._latest_timestamp = t_end
                    self._last_grab_time = t_end
                    self._recent_grab_times.append((t_end - t_start) * 1000.0)
                    if len(self._recent_grab_times) > 60:
                        self._recent_grab_times.pop(0)

                self._frame_ready_event.set()
                continue

            if not self.cap or not self.cap.isOpened():
                self._handle_reconnect()
                continue

            t_start = time.perf_counter()
            ret, frame = self.cap.read()
            t_end = time.perf_counter()

            if not ret or frame is None:
                logger.error("Error al leer frame de la cámara. Iniciando proceso de reconexión.")
                self.camera_healthy = False
                self._handle_reconnect()
                continue

            self.camera_healthy = True
            self._reconnect_backoff_s = 1.0  # Reset backoff on success

            # Filtrar frames negros vacíos de inicialización de DirectShow (muestreo rápido sin recorrer 6M de píxeles)
            if np.mean(frame[:20, :20]) == 0 and np.mean(frame[::16, ::16]) < 2.0:
                time.sleep(0.01)
                continue

            with self._lock:
                self.frames_captured += 1
                self._latest_frame = frame
                self._latest_timestamp = t_end

                # Calcular intervalo entre frames de hardware
                if self._last_grab_time > 0:
                    interval = (t_end - self._last_grab_time) * 1000.0
                    self._recent_intervals.append(interval)
                    if len(self._recent_intervals) > 60:
                        self._recent_intervals.pop(0)

                self._last_grab_time = t_end

                grab_latency = (t_end - t_start) * 1000.0
                self._recent_grab_times.append(grab_latency)
                if len(self._recent_grab_times) > 60:
                    self._recent_grab_times.pop(0)

            self._frame_ready_event.set()

    def _handle_reconnect(self):
        """Maneja el backoff exponencial e intento de reconexión de la cámara."""
        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info(f"Esperando {self._reconnect_backoff_s}s antes de intentar reconectar...")
        time.sleep(self._reconnect_backoff_s)
        
        if not self._is_running:
            return

        try:
            logger.info(f"Intentando reabrir cámara '{self.device_info.name}'...")
            self.cap = cv2.VideoCapture(self.device_info.index, self.device_info.backend)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Descartar iniciales para estabilizar exposición
                for _ in range(15):
                    self.cap.read()
                    
                self.camera_healthy = True
                self._reconnect_backoff_s = 1.0
                logger.info("Reconexión exitosa.")
                return
        except Exception as e:
            logger.error(f"Excepción durante reconexión: {e}")

        # Incrementar backoff
        self._reconnect_backoff_s = min(self._reconnect_backoff_s * 2, 8.0)

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Obtiene el frame más reciente.
        Retorna (exitoso, frame_bgr, latencia_de_lectura_ms).
        """
        t0 = time.perf_counter()
        if self.use_threaded_reader:
            if not self.camera_healthy:
                time.sleep(timeout)
                return False, None, 0.0

            if not self._frame_ready_event.wait(timeout=timeout):
                return False, None, 0.0

            with self._lock:
                if self._latest_frame is None:
                    return False, None, 0.0
                frame = self._latest_frame.copy()
                self.frames_delivered += 1
                self._frame_ready_event.clear()

            read_lat_ms = (time.perf_counter() - t0) * 1000.0
            return True, frame, read_lat_ms

        else:
            # Modo síncrono directo
            if self.cap is None or not self.cap.isOpened():
                return False, None, 0.0

            ret, frame = self.cap.read()
            t_end = time.perf_counter()
            lat_ms = (t_end - t0) * 1000.0

            if ret and frame is not None:
                self.frames_captured += 1
                self.frames_delivered += 1
                if self._last_grab_time > 0:
                    self._recent_intervals.append((t_end - self._last_grab_time) * 1000.0)
                    if len(self._recent_intervals) > 60:
                        self._recent_intervals.pop(0)
                self._last_grab_time = t_end
                self._recent_grab_times.append(lat_ms)
                if len(self._recent_grab_times) > 60:
                    self._recent_grab_times.pop(0)
                return True, frame, lat_ms

            return False, None, lat_ms

    def get_telemetry(self) -> Dict[str, Any]:
        """Retorna estadísticas en tiempo real del stream de cámara."""
        with self._lock:
            avg_grab_ms = (
                sum(self._recent_grab_times) / len(self._recent_grab_times)
                if self._recent_grab_times
                else 0.0
            )
            avg_interval_ms = (
                sum(self._recent_intervals) / len(self._recent_intervals)
                if self._recent_intervals
                else 0.0
            )
            effective_fps = 1000.0 / avg_interval_ms if avg_interval_ms > 0 else 0.0

            dropped = max(0, self.frames_captured - self.frames_delivered)

            return {
                "device_name": self.device_info.name,
                "device_index": self.device_info.index,
                "width": self.actual_width,
                "height": self.actual_height,
                "fourcc": self.fourcc_str,
                "reported_fps": self.reported_fps,
                "effective_fps": round(effective_fps, 1),
                "avg_grab_ms": round(avg_grab_ms, 2),
                "avg_interval_ms": round(avg_interval_ms, 2),
                "frames_captured": self.frames_captured,
                "frames_delivered": self.frames_delivered,
                "dropped_frames": dropped,
            }

    def close(self):
        """Cierra el stream y libera recursos."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Cámara C270 liberada correctamente.")


class MockCameraStream(CameraStream):
    """
    Clase derivada de CameraStream que fuerza la generación de cuadros sintéticos para pruebas sin hardware.
    """
    def __init__(
        self,
        device_info: Optional[CameraDeviceInfo] = None,
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: int = 30,
        use_threaded_reader: bool = True
    ):
        dev = device_info or CameraDeviceInfo(index=-1, name="Cámara Sintética de Diagnóstico", backend=-1, is_target=False)
        super().__init__(
            device_info=dev,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            use_threaded_reader=use_threaded_reader
        )

    def open(self) -> bool:
        self.is_synthetic = True
        self.actual_width = self.target_width
        self.actual_height = self.target_height
        self.reported_fps = float(self.target_fps)
        self.fourcc_str = "SYNT"
        self.camera_healthy = True
        if self.use_threaded_reader:
            self._is_running = True
            self._thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._thread.start()
            self._frame_ready_event.wait(timeout=2.0)
        return True

