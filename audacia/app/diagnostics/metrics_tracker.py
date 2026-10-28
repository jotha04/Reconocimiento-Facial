"""
Módulo de seguimiento y agregación de métricas de rendimiento en tiempo real
(latencias por componente, FPS efectivo y estabilidad de tiempos de frame).
Actualizado para Fase 4: Tracking, Attention, Detector y Captura.
"""

import time
from collections import deque
from typing import Dict, Any
import numpy as np


class MetricsTracker:
    """
    Registra muestras de tiempo de cada etapa del pipeline y calcula
    estadísticas acumuladas y ventanas rodantes.
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.capture_times = deque(maxlen=window_size)
        self.detector_times = deque(maxlen=window_size)
        self.tracking_times = deque(maxlen=window_size)
        self.attention_times = deque(maxlen=window_size)
        self.recognition_times = deque(maxlen=window_size)
        self.render_times = deque(maxlen=window_size)
        self.pipeline_times = deque(maxlen=window_size)
        self.frame_intervals = deque(maxlen=window_size)

        self._last_frame_timestamp = 0.0
        self.total_frames_processed = 0

    def record_frame_start(self) -> float:
        """Marca el inicio del procesamiento del frame y calcula el intervalo entre frames."""
        now = time.perf_counter()
        if self._last_frame_timestamp > 0:
            interval = (now - self._last_frame_timestamp) * 1000.0
            self.frame_intervals.append(interval)
        self._last_frame_timestamp = now
        self.total_frames_processed += 1
        return now

    def record_metrics(
        self,
        capture_ms: float,
        detector_ms: float,
        tracking_ms: float,
        attention_ms: float,
        render_ms: float,
        pipeline_ms: float,
        recognition_ms: float = 0.0
    ):
        """Almacena las mediciones de latencia del ciclo actual."""
        self.capture_times.append(capture_ms)
        self.detector_times.append(detector_ms)
        self.tracking_times.append(tracking_ms)
        self.attention_times.append(attention_ms)
        self.recognition_times.append(recognition_ms)
        self.render_times.append(render_ms)
        self.pipeline_times.append(pipeline_ms)

    def get_summary(self) -> Dict[str, Any]:
        """Calcula promedios y métricas estadísticas representativas."""
        def safe_stats(d: deque):
            if not d:
                return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            arr = np.array(d)
            return {
                "avg": round(float(np.mean(arr)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
                "std": round(float(np.std(arr)), 2),
            }

        pipe_stats = safe_stats(self.pipeline_times)
        int_stats = safe_stats(self.frame_intervals)

        effective_fps = (
            round(1000.0 / int_stats["avg"], 1) if int_stats["avg"] > 0 else 0.0
        )
        max_theoretical_fps = (
            round(1000.0 / pipe_stats["avg"], 1) if pipe_stats["avg"] > 0 else 0.0
        )

        return {
            "total_frames": self.total_frames_processed,
            "effective_fps": effective_fps,
            "max_theoretical_fps": max_theoretical_fps,
            "capture_ms": safe_stats(self.capture_times),
            "detector_ms": safe_stats(self.detector_times),
            "tracking_ms": safe_stats(self.tracking_times),
            "attention_ms": safe_stats(self.attention_times),
            "recognition_ms": safe_stats(self.recognition_times),
            "render_ms": safe_stats(self.render_times),
            "pipeline_ms": pipe_stats,
            "frame_interval_ms": int_stats
        }
