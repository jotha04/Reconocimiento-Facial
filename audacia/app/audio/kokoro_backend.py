"""
Backend de Síntesis de Voz Neuronal Kokoro-82M (KokoroBackend).
Encapsula la inferencia ONNX con la voz 'ef_dora' en español.
Implementa BaseTTSBackend para garantizar desacoplamiento y sustituibilidad.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseTTSBackend(ABC):
    """Interfaz abstracta base para motores de síntesis de voz locales."""

    @abstractmethod
    def initialize(self) -> bool:
        """Carga modelos y dependencias en memoria. Retorna True si tuvo éxito."""
        pass

    @abstractmethod
    def synthesize_to_file(self, normalized_text: str, output_wav_path: str) -> bool:
        """Sintetiza texto normalizado a un archivo WAV local."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Indica si el motor está inicializado y listo para sintetizar."""
        pass


class KokoroBackend(BaseTTSBackend):
    """
    Backend basado en Kokoro-82M ONNX (kokoro-onnx 0.6.1).
    Utiliza el banco de voces binario voices-v1.0.bin y fonemización con espeak-ng.
    """

    def __init__(
        self,
        model_path: str = "scratch/models/kokoro/kokoro-v1.0.onnx",
        voices_path: str = "scratch/models/kokoro/voices-v1.0.bin",
        voice: str = "ef_dora",
        language: str = "es",
        model_version: str = "kokoro-v1.0"
    ):
        # Resolución flexible y portable de rutas de modelos y voces
        resolved_model = model_path
        if not os.path.isfile(resolved_model):
            alt1 = os.path.join("models", "tts", "kokoro", os.path.basename(model_path))
            alt2 = os.path.join("scratch", "models", "kokoro", os.path.basename(model_path))
            if os.path.isfile(alt1):
                resolved_model = alt1
            elif os.path.isfile(alt2):
                resolved_model = alt2

        resolved_voices = voices_path
        if not os.path.isfile(resolved_voices):
            alt1 = os.path.join("models", "tts", "kokoro", os.path.basename(voices_path))
            alt2 = os.path.join("scratch", "models", "kokoro", os.path.basename(voices_path))
            if os.path.isfile(alt1):
                resolved_voices = alt1
            elif os.path.isfile(alt2):
                resolved_voices = alt2

        self.model_path = os.path.abspath(resolved_model)
        self.voices_path = os.path.abspath(resolved_voices)
        self.voice = voice
        self.language = language
        self.model_version = model_version

        self._kokoro = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._kokoro is not None

    def initialize(self) -> bool:
        """
        Carga el modelo ONNX y el archivo de voces en memoria.
        Diseñado para invocarse bajo demanda (lazy-load) o en arranque.
        """
        if self._is_ready:
            return True

        if not os.path.isfile(self.model_path):
            logger.error(f"[KokoroBackend] Archivo de modelo no encontrado: {self.model_path}")
            return False

        if not os.path.isfile(self.voices_path):
            logger.error(f"[KokoroBackend] Archivo de voces no encontrado: {self.voices_path}")
            return False

        try:
            import kokoro_onnx
            import onnxruntime as rt
            from kokoro_onnx.session import resolve_providers

            logger.info(f"[KokoroBackend] Inicializando grafo ONNX desde {self.model_path}...")
            sess_opts = rt.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 1
            sess_opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
            session = rt.InferenceSession(self.model_path, sess_options=sess_opts, providers=resolve_providers())
            self._kokoro = kokoro_onnx.Kokoro.from_session(session, self.voices_path)
            self._is_ready = True
            logger.info(f"[KokoroBackend] Inicializado exitosamente (Voz: '{self.voice}', Idioma: '{self.language}')")
            return True
        except Exception as e:
            logger.error(f"[KokoroBackend] Falla crítica al inicializar kokoro-onnx: {e}", exc_info=True)
            self._kokoro = None
            self._is_ready = False
            return False

    def synthesize_to_file(self, normalized_text: str, output_wav_path: str) -> bool:
        """
        Sintetiza texto normalizado a archivo WAV (24000 Hz, 16-bit PCM).
        Retorna True si el archivo fue generado correctamente.
        """
        if not self._is_ready:
            ok = self.initialize()
            if not ok:
                return False

        try:
            import soundfile as sf

            os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
            # Inferencia de red neuronal acústica y vocoder
            samples, sr = self._kokoro.create(
                normalized_text,
                voice=self.voice,
                speed=1.0,
                lang=self.language
            )

            # Escritura directa a archivo WAV PCM
            sf.write(output_wav_path, samples, sr)

            if os.path.isfile(output_wav_path) and os.path.getsize(output_wav_path) > 44:
                return True

            logger.error(f"[KokoroBackend] Archivo de salida inválido o vacío: {output_wav_path}")
            return False

        except Exception as e:
            logger.error(f"[KokoroBackend] Error durante síntesis de '{normalized_text}': {e}", exc_info=True)
            return False
