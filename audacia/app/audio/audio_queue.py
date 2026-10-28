"""
Cola y Reproductor de Audio Asíncrono (AudioQueue).
Garantiza:
1. No superposición de audios (reproducción secuencial y controlada).
2. Ejecución 100% no bloqueante: la cámara y visión siguen corriendo a 30 FPS en CPU.
3. Telemetría completa de latencia de síntesis y duración de locución.
"""

import os
import sys
import time
import queue
import logging
import threading
from typing import Optional, Callable, Dict, Any

from audacia.app.audio.tts_engine import TTSEngine
from audacia.app.attention.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioQueue:
    """
    Gestiona una cola de eventos de audio con un hilo de trabajo en segundo plano.
    """

    def __init__(self, config: Optional[AudioConfig] = None, tts_engine: Optional[TTSEngine] = None):
        self.config = config or AudioConfig()
        self.tts = tts_engine or TTSEngine(self.config)

        self._queue = queue.Queue(maxsize=4)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._is_playing = False
        self._current_text = ""
        self._current_audio_process = None

        # Métricas y telemetría
        self.greetings_played_count = 0
        self.last_synthesis_ms = 0.0
        self.last_playback_duration_s = 0.0

        # Directorio temporal para audios sintetizados
        self.audio_tmp_dir = "data/audio_cache"
        os.makedirs(self.audio_tmp_dir, exist_ok=True)

        if self.config.enabled:
            self.start()

    def start(self):
        """Inicia el hilo de trabajo de audio."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="AudacIA-AudioWorker")
        self._worker_thread.start()
        logger.info("AudioQueue worker thread iniciado.")

    def stop(self):
        """Detiene el hilo de trabajo de audio y termina reproducciones en curso."""
        self._running = False
        if self._current_audio_process:
            try:
                self._current_audio_process.terminate()
                try:
                    self._current_audio_process.wait(timeout=0.5)
                except Exception:
                    self._current_audio_process.kill()
            except Exception as e:
                logger.error(f"Error terminando proceso de audio: {e}")
                
        self._queue.put(None) # Señal de salida
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        logger.info("AudioQueue detenido.")

    def is_playing(self) -> bool:
        """Indica si hay un audio en proceso de síntesis o reproducción."""
        return self._is_playing

    def get_current_text(self) -> str:
        """Texto del saludo que se está reproduciendo actualmente."""
        return self._current_text if self._is_playing else ""

    def enqueue_greeting(
        self,
        text: str,
        on_started: Optional[Callable[[str], None]] = None,
        on_completed: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Encola un saludo para ser sintetizado y reproducido.
        Si ya hay un audio en reproducción o la cola está ocupada, se rechaza
        para evitar superposiciones no deseadas.
        """
        if not self.config.enabled:
            return False

        if self._is_playing:
            logger.debug(f"Audio ocupado. Descartando solicitud de saludo: '{text}'")
            return False

        try:
            self._queue.put_nowait({
                "text": text,
                "on_started": on_started,
                "on_completed": on_completed,
                "enqueued_at": time.monotonic()
            })
            logger.info(f"Saludo encolado con éxito: '{text}'")
            return True
        except queue.Full:
            logger.warning(f"Cola de audio llena. Descartando '{text}'")
            return False

    def _worker_loop(self):
        """Bucle continuo del hilo de reproducción de audio."""
        # Inicializar COM en Windows para el hilo secundario
        if sys.platform == "win32":
            try:
                import comtypes
                comtypes.CoInitialize()
            except Exception:
                pass

        try:
            self.tts.initialize()
        except Exception as e:
            logger.error(f"Falla inicializando TTS en worker: {e}")

        while self._running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None or not self._running:
                break

            text = item["text"]
            on_started = item.get("on_started")
            on_completed = item.get("on_completed")

            self._is_playing = True
            self._current_text = text

            playback_success = False
            wav_path = None
            cleanup_after = False

            try:
                if on_started:
                    try:
                        on_started(text)
                    except Exception as e:
                        logger.error(f"Error en callback on_started: {e}")

                # 1. Obtención/Síntesis del artefacto de audio (con soporte de caché y normalización)
                t_synth_0 = time.perf_counter()
                artifact = self.tts.get_audio_artifact(text)
                t_synth_1 = time.perf_counter()
                self.last_synthesis_ms = (t_synth_1 - t_synth_0) * 1000.0

                wav_path = artifact.wav_path if artifact else None
                cleanup_after = artifact.cleanup_after_playback if artifact else False

                # 2. Reproducción de audio en el hilo secundario
                t_play_0 = time.perf_counter()
                if wav_path and os.path.isfile(wav_path):
                    self._play_wav_file(wav_path)
                    playback_success = True
                else:
                    logger.warning(f"No se pudo generar o recuperar audio para '{text}'")

                t_play_1 = time.perf_counter()
                self.last_playback_duration_s = t_play_1 - t_play_0
                self.greetings_played_count += 1

                backend_info = artifact.backend_used if artifact else "NONE"
                logger.info(
                    f"Saludo completado: '{text}' | "
                    f"Resolución/Síntesis: {self.last_synthesis_ms:.1f}ms | Duración: {self.last_playback_duration_s:.2f}s | "
                    f"Backend: {backend_info}"
                )

            except Exception as e:
                logger.error(f"Error reproduciendo saludo '{text}': {e}")
                playback_success = False

            finally:
                # BUG-003 fix: SIEMPRE llamar a on_completed, sea éxito o fallo
                if on_completed:
                    try:
                        on_completed(text, success=playback_success)
                    except TypeError:
                        try:
                            on_completed(text)
                        except Exception as e:
                            logger.error(f"Error en callback on_completed: {e}")
                    except Exception as e:
                        logger.error(f"Error en callback on_completed: {e}")

                # Limpieza segura: ÚNICAMENTE si cleanup_after_playback es True (temporales / fallback)
                if cleanup_after and wav_path and os.path.isfile(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                self._is_playing = False
                self._current_text = ""
                self._queue.task_done()

        if sys.platform == "win32":
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass

    def _play_wav_file(self, wav_path: str):
        """Reproduce un archivo WAV de forma síncrona, soportando cancelación (Popen)."""
        if sys.platform == "win32":
            import winsound
            # En Windows SND_SYNC es síncrono.
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        else:
            import subprocess
            cmd = ["afplay", wav_path] if sys.platform == "darwin" else ["aplay", wav_path]
            try:
                self._current_audio_process = subprocess.Popen(cmd)
                self._current_audio_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout al reproducir audio {wav_path}. Terminando proceso.")
                self._current_audio_process.terminate()
                self._current_audio_process.wait(timeout=1.0)
            except Exception as e:
                logger.error(f"Excepción reproduciento audio con Popen: {e}")
                if self._current_audio_process:
                    self._current_audio_process.terminate()
            finally:
                self._current_audio_process = None
