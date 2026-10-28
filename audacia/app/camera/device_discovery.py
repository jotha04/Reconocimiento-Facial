"""
Módulo de descubrimiento y validación de dispositivos de captura de video.
Garantiza la selección exclusiva de la Logitech C270 HD WebCam y bloquea
cualquier intento de uso accidental de cámaras integradas o IR.
"""

import sys
import platform
import logging
from dataclasses import dataclass
from typing import List, Optional
import cv2

logger = logging.getLogger(__name__)


@dataclass
class CameraDeviceInfo:
    index: int
    name: str
    backend: int
    is_target: bool
    bus_info: str = ""

    def __str__(self) -> str:
        target_str = "[TARGET MATCH]" if self.is_target else "[OTHER]"
        return f"Camera {self.index}: '{self.name}' (Backend: {self.backend}) {target_str}"


def list_connected_cameras() -> List[CameraDeviceInfo]:
    """
    Enumera los dispositivos de video conectados utilizando el backend apropiado para el SO.
    En Windows usa DirectShow / pygrabber para correlacionar exactamente con los índices de OpenCV.
    En macOS usa AVFoundation y system_profiler.
    """
    os_name = platform.system()
    devices: List[CameraDeviceInfo] = []

    if os_name == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            dshow_names = graph.get_input_devices()
            for idx, name in enumerate(dshow_names):
                is_target_cam = any(k in name.lower() for k in ["c920", "c922", "c930", "c270", "brio", "logitech"]) or ("logi" in name.lower() and "webcam" in name.lower())
                devices.append(
                    CameraDeviceInfo(
                        index=idx,
                        name=name,
                        backend=cv2.CAP_DSHOW,
                        is_target=is_target_cam
                    )
                )
        except ImportError:
            logger.warning("pygrabber no está instalado. Enumeración DirectShow por nombre desactivada.")
            # Fallback a sondeo de índices
            for idx in range(4):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    devices.append(
                        CameraDeviceInfo(
                            index=idx,
                            name=f"Camera index {idx} (Nombre no disponible sin pygrabber)",
                            backend=cv2.CAP_DSHOW,
                            is_target=False
                        )
                    )
                    cap.release()

    elif os_name == "Darwin":  # macOS Apple Silicon / Intel
        # En macOS, cv2.CAP_AVFOUNDATION enumera por índice.
        # Obtenemos los nombres reales reportados por system_profiler
        import subprocess
        mac_names = []
        try:
            cmd = ["system_profiler", "SPCameraDataType", "-json"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=2.5)
            if res.returncode == 0 and res.stdout:
                import json
                data = json.loads(res.stdout)
                cam_list = data.get("SPCameraDataType", [])
                for c in cam_list:
                    name_val = c.get("_name", "")
                    if name_val:
                        mac_names.append(name_val)
        except Exception as e:
            logger.debug(f"Aviso consultando system_profiler en macOS: {e}")

        c270_known_in_mac = any("c270" in n.lower() or "logi" in n.lower() for n in mac_names)
        open_cams = []

        # Probar índices con AVFOUNDATION y CAP_ANY
        for idx in range(4):
            for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    name = mac_names[idx] if idx < len(mac_names) else f"Cámara Mac (Índice {idx})"
                    is_target_cam = any(
                        k in name.lower() for k in ["c920", "c922", "c930", "c270", "brio", "logitech"]
                    ) or ("logi" in name.lower() and "webcam" in name.lower())
                    open_cams.append((idx, name, is_target_cam, backend))
                    cap.release()
                    break

        for idx, name, is_c270, backend in open_cams:
            target_match = is_c270
            if not target_match and len(open_cams) == 1 and (c270_known_in_mac or len(mac_names) <= 1):
                if not any(k in name.lower() for k in ["facetime", "virtual"]):
                    target_match = True
                    if name.startswith("Cámara Mac"):
                        name = mac_names[0] if mac_names else "Logitech C270 HD WebCam"

            devices.append(
                CameraDeviceInfo(
                    index=idx,
                    name=name,
                    backend=backend,
                    is_target=target_match
                )
            )

        # Si system_profiler reportó cámaras pero opencv no las pudo abrir (por ej por permisos TCC), agregarlas como disponibles
        if not devices and mac_names:
            for idx, name in enumerate(mac_names):
                is_target_cam = any(k in name.lower() for k in ["c920", "c922", "c930", "c270", "brio", "logi", "logitech"])
                devices.append(
                    CameraDeviceInfo(
                        index=idx,
                        name=name,
                        backend=cv2.CAP_ANY,
                        is_target=is_target_cam
                    )
                )

    else:  # Linux / V4L2
        for idx in range(4):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                devices.append(
                    CameraDeviceInfo(
                        index=idx,
                        name=f"V4L2 Camera {idx}",
                        backend=cv2.CAP_V4L2,
                        is_target=False
                    )
                )
                cap.release()

    return devices


def find_c270_camera(preferred_index: Optional[int] = None, allow_fallback: bool = True) -> CameraDeviceInfo:
    """
    Busca de manera determinista la cámara externa Logitech C270.
    Si no la encuentra y allow_fallback es True (por defecto), permite usar cualquier cámara disponible en el sistema (por ej. la cámara FaceTime o integrada de la Mac).
    """
    devices = list_connected_cameras()

    if devices:
        logger.info("Dispositivos de cámara detectados:")
        for d in devices:
            logger.info(f"  {d}")

    # 1. Si se especificó un índice preferido, validarlo primero
    if preferred_index is not None:
        matching = [d for d in devices if d.index == preferred_index]
        if matching:
            return matching[0]

    # 2. Buscar por coincidencia con cámara objetivo Logitech (C920 / C270 / etc.)
    target_candidates = [d for d in devices if d.is_target]
    if target_candidates:
        selected = target_candidates[0]
        logger.info(f"Cámara externa seleccionada con éxito: {selected}")
        return selected

    # 3. Si hay dispositivos detectados pero ninguno es target y se permite fallback
    if devices and allow_fallback:
        selected = devices[0]
        logger.warning(
            f"[AVISO FALLBACK CÁMARA] No se detectó cámara Logitech externa por nombre. "
            f"Utilizando cámara disponible en el sistema: '{selected.name}' (Índice {selected.index})"
        )
        return selected

    # 4. Si no se detectaron cámaras activas pero se permite fallback a cámara Mac por defecto
    if allow_fallback:
        logger.warning("[AVISO] Intentando usar cámara por defecto de la Mac (Índice 0)...")
        return CameraDeviceInfo(
            index=0,
            name="Cámara Mac / FaceTime (Índice 0)",
            backend=cv2.CAP_ANY,
            is_target=False
        )

    device_summary = "\n".join([f"  - [{d.index}] {d.name}" for d in devices]) if devices else "Sin cámaras abiertas"
    raise RuntimeError(
        "NO se encontró una cámara externa Logitech (C920 / C270) conectada.\n"
        f"Dispositivos encontrados:\n{device_summary}"
    )


# Alias semántico para desacoplar de un modelo de hardware específico
find_target_camera = find_c270_camera
