"""
Gestor de Caché Persistente de Audio (AudioCacheManager).
Responsable exclusivo de:
- Cálculo determinista de claves de caché (SHA256 con control de versiones).
- Búsqueda (lookup) sin carga de bytes innecesaria.
- Validación física de archivos WAV mediante módulo nativo 'wave'.
- Escritura atómica a disco para evitar archivos corruptos ante interrupciones.
"""

import os
import time
import wave
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AudioCacheManager:
    """
    Administra la persistencia, indexación y validación de archivos WAV cacheados.
    """

    DEFAULT_CACHE_VERSION = "v1"

    def __init__(
        self,
        cache_dir: str = "data/audio_cache",
        cache_version: str = DEFAULT_CACHE_VERSION,
        enabled: bool = True
    ):
        self.cache_dir = os.path.abspath(cache_dir)
        self.cache_version = cache_version
        self.enabled = enabled

        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    def compute_key(
        self,
        normalized_text: str,
        backend: str,
        voice: str,
        model_version: str,
        language: str = "es",
        normalizer_version: str = "v1"
    ) -> str:
        """
        Calcula una clave hash SHA-256 única y estable.
        Incluye todas las dimensiones que pueden alterar el audio físico o la fonética.
        """
        payload = (
            f"{self.cache_version}|"
            f"{backend.strip().lower()}|"
            f"{model_version.strip().lower()}|"
            f"{voice.strip().lower()}|"
            f"{language.strip().lower()}|"
            f"{normalizer_version.strip().lower()}|"
            f"{normalized_text.strip()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_cached_path(self, key: str) -> Optional[str]:
        """
        Comprueba la existencia y validez del archivo en caché.
        Retorna la ruta absoluta si existe y es un WAV íntegro; de lo contrario retorna None.
        NO lee el contenido completo del archivo a memoria (0.0 ms sobrecosto).
        """
        if not self.enabled:
            return None

        target_path = os.path.join(self.cache_dir, f"{key}.wav")
        if not os.path.isfile(target_path):
            return None

        # Validación estructural rápida del formato WAV
        if self.is_valid_wav(target_path):
            return target_path

        # Si el archivo está corrupto o truncado, purgarlo
        logger.warning(f"[AudioCache] Archivo corrupto detectado en caché: {target_path}. Purgando.")
        try:
            os.remove(target_path)
        except OSError as e:
            logger.error(f"[AudioCache] No se pudo eliminar archivo corrupto: {e}")

        return None

    def is_valid_wav(self, file_path: str) -> bool:
        """
        Valida que el archivo exista, tenga un tamaño razonable (>44 bytes de cabecera)
        y que el contenedor RIFF/WAVE sea sintácticamente correcto.
        """
        try:
            if not os.path.isfile(file_path):
                return False
            if os.path.getsize(file_path) <= 44:
                return False

            with wave.open(file_path, "rb") as wf:
                channels = wf.getnchannels()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                return channels > 0 and framerate > 0 and nframes > 0
        except Exception:
            return False

    def save_atomic(self, key: str, temp_wav_path: str) -> Optional[str]:
        """
        Guarda de forma atómica y segura un archivo temporal en la ruta final de caché.
        Valida la integridad del WAV antes de ejecutar el reemplazo atómico con os.replace.
        Retorna la ruta final persistida si es exitoso; None si falló la validación.
        """
        if not self.enabled:
            return None

        if not self.is_valid_wav(temp_wav_path):
            logger.error(f"[AudioCache] Intento de cachear un WAV inválido desde {temp_wav_path}")
            try:
                if os.path.isfile(temp_wav_path):
                    os.remove(temp_wav_path)
            except OSError:
                pass
            return None

        final_path = os.path.join(self.cache_dir, f"{key}.wav")
        try:
            # os.replace es atómico en POSIX y Windows (sobrescribe limpiamente)
            os.replace(temp_wav_path, final_path)
            return final_path
        except OSError as e:
            logger.error(f"[AudioCache] Fallo al mover atómicamente {temp_wav_path} a {final_path}: {e}")
            try:
                if os.path.isfile(temp_wav_path):
                    os.remove(temp_wav_path)
            except OSError:
                pass
            return None
