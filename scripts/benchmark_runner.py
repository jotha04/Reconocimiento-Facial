"""
Herramienta de benchmark automatizado para medir empíricamente y documentar
el rendimiento real del pipeline de visión de AudacIA sobre la C270.
"""

import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import logging
from typing import Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("AudacIA-Benchmark")

from audacia.app.camera.device_discovery import find_c270_camera
from audacia.app.camera.camera_stream import CameraStream
from audacia.app.detection.yunet_detector import YuNetDetector
from audacia.app.motion.camera_motion import CameraMotionDetector
from audacia.app.diagnostics.metrics_tracker import MetricsTracker
from audacia.app.diagnostics.visualizer import DiagnosticVisualizer


def run_benchmark(
    target_frames: int = 180,
    width: int = 1920,
    height: int = 1080,
    motion_method: str = "sparse_flow",
    min_face_size: int = 0
) -> Dict[str, Any]:
    print("=" * 65)
    print(f"  EJECUTANDO BENCHMARK AUDACIA ({target_frames} frames @ {width}x{height})")
    print("=" * 65)

    c270_info = find_c270_camera()
    camera = CameraStream(
        device_info=c270_info,
        target_width=width,
        target_height=height,
        target_fps=30,
        use_threaded_reader=True
    )

    detector = YuNetDetector(
        model_path="models/face_detection_yunet_2023mar.onnx",
        input_size=(width, height),
        score_threshold=0.6,
        nms_threshold=0.3,
        min_face_size=min_face_size,
        max_inference_dimension=1280
    )

    motion_detector = CameraMotionDetector(method=motion_method)
    metrics = MetricsTracker(window_size=target_frames)
    visualizer = DiagnosticVisualizer()

    camera.open()
    time.sleep(0.5)

    detected_faces_sizes = []
    detected_faces_confidences = []
    face_counts_histogram = {0: 0, 1: 0, 2: 0, "3+": 0}
    motion_states_count = {"CAMERA_STABLE": 0, "CAMERA_MOVING": 0}

    frames_processed = 0
    t_bench_start = time.perf_counter()

    try:
        while frames_processed < target_frames:
            t_f0 = metrics.record_frame_start()

            # 1. Captura
            ret, frame, cap_ms = camera.read(timeout=0.5)
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # 2. Detección
            faces, det_ms = detector.detect(frame)

            # 3. Movimiento
            motion_telemetry = motion_detector.update(frame)
            mot_ms = motion_telemetry.calc_time_ms

            # 4. Render
            t_ren0 = time.perf_counter()
            cam_telemetry = camera.get_telemetry()
            summary = metrics.get_summary()
            _ = visualizer.render(frame, faces, motion_telemetry, summary, cam_telemetry)
            ren_ms = (time.perf_counter() - t_ren0) * 1000.0

            t_pipe_end = time.perf_counter()
            pipe_ms = (t_pipe_end - t_f0) * 1000.0

            metrics.record_metrics(cap_ms, det_ms, mot_ms, ren_ms, pipe_ms)

            # Recolectar datos de rostros
            num_faces = len(faces)
            if num_faces == 0:
                face_counts_histogram[0] += 1
            elif num_faces == 1:
                face_counts_histogram[1] += 1
            elif num_faces == 2:
                face_counts_histogram[2] += 1
            else:
                face_counts_histogram["3+"] += 1

            for f in faces:
                detected_faces_sizes.append({"w": f.w, "h": f.h, "area": f.area})
                detected_faces_confidences.append(f.confidence)

            # Recolectar estados de movimiento
            motion_states_count[motion_telemetry.state.value] += 1

            frames_processed += 1
            if frames_processed % 30 == 0:
                print(f"  Progreso: {frames_processed}/{target_frames} frames procesados...")

    finally:
        total_bench_duration = time.perf_counter() - t_bench_start
        camera.close()

    cam_info = camera.get_telemetry()
    final_metrics = metrics.get_summary()

    real_fps = round(frames_processed / total_bench_duration, 1)

    results = {
        "hardware": {
            "device_name": cam_info["device_name"],
            "device_index": cam_info["device_index"],
            "resolution_requested": f"{width}x{height}",
            "resolution_actual": f"{cam_info['width']}x{cam_info['height']}",
            "pixel_format": cam_info["fourcc"],
            "reported_fps": cam_info["reported_fps"]
        },
        "performance": {
            "total_frames": frames_processed,
            "total_duration_seconds": round(total_bench_duration, 2),
            "effective_fps_measured": real_fps,
            "capture_latency_ms": final_metrics["capture_ms"],
            "detector_latency_ms": final_metrics["detector_ms"],
            "motion_latency_ms": final_metrics["motion_ms"],
            "render_latency_ms": final_metrics["render_ms"],
            "pipeline_total_ms": final_metrics["pipeline_ms"],
            "frame_interval_ms": final_metrics["frame_interval_ms"],
            "camera_dropped_frames": cam_info["dropped_frames"]
        },
        "detection_data": {
            "score_threshold": detector.score_threshold,
            "min_face_size": detector.min_face_size,
            "face_counts_distribution": face_counts_histogram,
            "total_face_instances_observed": len(detected_faces_sizes),
            "face_confidences": {
                "avg": round(float(np.mean(detected_faces_confidences)), 3) if detected_faces_confidences else 0.0,
                "min": round(float(np.min(detected_faces_confidences)), 3) if detected_faces_confidences else 0.0,
                "max": round(float(np.max(detected_faces_confidences)), 3) if detected_faces_confidences else 0.0
            }
        },
        "motion_data": {
            "method": motion_method,
            "states_distribution": motion_states_count
        }
    }

    os.makedirs("data", exist_ok=True)
    out_file = f"data/benchmark_results_{width}x{height}_{motion_method}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("                RESULTADOS DEL BENCHMARK REAL                ")
    print("=" * 65)
    print(f"Cámara: {results['hardware']['device_name']} [{results['hardware']['resolution_actual']} | {results['hardware']['pixel_format']}]")
    print(f"Frames analizados: {frames_processed} en {total_bench_duration:.2f} s")
    print(f"FPS REAL EFECTIVO: {real_fps} FPS (Reportado por API: {cam_info['reported_fps']:.1f})")
    print("-" * 65)
    print("LATENCIAS DESGLOSADAS (Promedio / Mín / Máx / Desv.Est):")
    print(f"  Captura (Grab/Read)  : {final_metrics['capture_ms']['avg']} ms (min: {final_metrics['capture_ms']['min']}, max: {final_metrics['capture_ms']['max']})")
    print(f"  Detector (YuNet ONNX): {final_metrics['detector_ms']['avg']} ms (min: {final_metrics['detector_ms']['min']}, max: {final_metrics['detector_ms']['max']})")
    print(f"  Movimiento ({motion_method}): {final_metrics['motion_ms']['avg']} ms (min: {final_metrics['motion_ms']['min']}, max: {final_metrics['motion_ms']['max']})")
    print(f"  Renderizado (GUI/HUD): {final_metrics['render_ms']['avg']} ms (min: {final_metrics['render_ms']['min']}, max: {final_metrics['render_ms']['max']})")
    print(f"  Total Pipeline       : {final_metrics['pipeline_ms']['avg']} ms (min: {final_metrics['pipeline_ms']['min']}, max: {final_metrics['pipeline_ms']['max']})")
    print("-" * 65)
    print(f"Frames descartados por buffer: {cam_info['dropped_frames']}")
    print(f"Distribución de rostros: {face_counts_histogram}")
    print(f"Reporte JSON guardado en: {out_file}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_benchmark(target_frames=180, width=1920, height=1080, motion_method="sparse_flow")
