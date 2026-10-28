"""
Motor Local de Síntesis de Voz (Local TTSEngine) — Fase 7.
Coordina:
1. Normalización fonética superficial (TTSTextNormalizer).
2. Caché persistente en disco con validación de integridad (AudioCacheManager).
3. Backend neuronal Kokoro-82M (KokoroBackend) con lazy-loading.
4. Fallback transparente y seguro al sintetizador nativo (pyttsx3/SAPI5/NSSpeech).
5. Devolución de AudioArtifact con semántica estricta de ciclo de vida (cleanup_after_playback).
"""

import os
import sys
import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from audacia.app.attention.config import AudioConfig
from audacia.app.audio.normalizer import TTSTextNormalizer
from audacia.app.audio.cache_manager import AudioCacheManager
from audacia.app.audio.kokoro_backend import KokoroBackend

logger = logging.getLogger(__name__)


@dataclass
class AudioArtifact:
    """
    Representa el resultado de la resolución de audio para un saludo.
    Encapsula inequívocamente la propiedad y política de limpieza del archivo.
    """
    wav_path: str
    cleanup_after_playback: bool = False
    backend_used: str = "kokoro"
    is_cached: bool = True
    normalized_text: str = ""


class TTSEngine:
    """
    Sintetizador de voz modular con soporte de caché persistente y fallback nativo.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.normalizer = TTSTextNormalizer(
            overrides=getattr(self.config, "pronunciation_overrides", None)
        )
        self.cache_manager = AudioCacheManager(
            cache_dir=self.config.cache_dir,
            enabled=self.config.cache_enabled
        )

        self._kokoro_backend: Optional[KokoroBackend] = None
        self._synthesis_lock = threading.Lock()
        self._is_initialized = False

    def initialize(self):
        """Inicializa los recursos base del motor y prepara los backends."""
        if self._is_initialized:
            return

        if sys.platform == "win32":
            try:
                import comtypes
                comtypes.CoInitialize()
            except Exception as e:
                logger.debug(f"[TTSEngine] comtypes CoInitialize: {e}")

        # Configurar backend Kokoro si está seleccionado
        if self.config.tts_engine.lower() == "kokoro":
            self._kokoro_backend = KokoroBackend(
                model_path=self.config.kokoro_model_path,
                voices_path=self.config.kokoro_voices_path,
                voice=self.config.kokoro_voice,
                language=self.config.kokoro_language,
                model_version=self.config.kokoro_model_version
            )
            # Si lazy_load es False, precargar en memoria inmediatamente
            if not self.config.lazy_load:
                self._kokoro_backend.initialize()

        self._is_initialized = True
        logger.info(f"[TTSEngine] Inicializado (Motor: {self.config.tts_engine}, Caché: {self.config.cache_enabled})")

    def get_audio_artifact(self, logical_text: str) -> Optional[AudioArtifact]:
        """
        Método principal de resolución de audio:
        1. Normaliza el texto lógico para TTS.
        2. Consulta la existencia en el AudioCacheManager.
        3. En CACHE HIT: Retorna AudioArtifact persistente (cleanup_after_playback=False).
        4. En CACHE MISS: Sintetiza con Kokoro, guarda atómicamente en caché y retorna persistente.
        5. En FALLO de Kokoro: Conmuta al backend nativo y retorna temporal (cleanup_after_playback=True).
        """
        if not self._is_initialized:
            self.initialize()

        # 1. Normalización superficial (preserva texto lógico afuera)
        norm_text = self.normalizer.normalize(logical_text)

        # 2. Clave única de caché
        key = self.cache_manager.compute_key(
            normalized_text=norm_text,
            backend=self.config.tts_engine,
            voice=self.config.kokoro_voice,
            model_version=self.config.kokoro_model_version,
            language=self.config.kokoro_language,
            normalizer_version=self.normalizer.version
        )

        # 3. Comprobación en caché (CACHE HIT)
        cached_wav = self.cache_manager.get_cached_path(key)
        if cached_wav:
            logger.info(f"[TTS] Cache HIT ({key[:12]}...) para '{logical_text}'")
            return AudioArtifact(
                wav_path=cached_wav,
                cleanup_after_playback=False,
                backend_used="cache",
                is_cached=True,
                normalized_text=norm_text
            )

        # 4. CACHE MISS — Síntesis protegida por lock
        logger.info(f"[TTS] Cache MISS ({key[:12]}...) para '{logical_text}' -> Normalizado: '{norm_text}'")
        with self._synthesis_lock:
            # Doble comprobación post-lock
            cached_wav = self.cache_manager.get_cached_path(key)
            if cached_wav:
                return AudioArtifact(
                    wav_path=cached_wav,
                    cleanup_after_playback=False,
                    backend_used="cache",
                    is_cached=True,
                    normalized_text=norm_text
                )

            # Intentar síntesis con Kokoro si es el motor seleccionado
            if self.config.tts_engine.lower() == "kokoro":
                if self._kokoro_backend is None:
                    self._kokoro_backend = KokoroBackend(
                        model_path=self.config.kokoro_model_path,
                        voices_path=self.config.kokoro_voices_path,
                        voice=self.config.kokoro_voice,
                        language=self.config.kokoro_language,
                        model_version=self.config.kokoro_model_version
                    )

                temp_path = os.path.join(
                    self.cache_manager.cache_dir,
                    f".tmp_{key[:16]}_{os.getpid()}_{int(time.monotonic() * 1000)}.wav"
                )

                kokoro_ok = self._kokoro_backend.synthesize_to_file(norm_text, temp_path)
                if kokoro_ok and os.path.isfile(temp_path):
                    if self.cache_manager.enabled:
                        final_path = self.cache_manager.save_atomic(key, temp_path)
                        if final_path:
                            logger.info(f"[TTS] Kokoro sintetizado y cacheado exitosamente: {final_path}")
                            return AudioArtifact(
                                wav_path=final_path,
                                cleanup_after_playback=False,
                                backend_used="kokoro",
                                is_cached=True,
                                normalized_text=norm_text
                            )
                    else:
                        # Caché desactivada: conservar temp y marcar para borrado
                        return AudioArtifact(
                            wav_path=temp_path,
                            cleanup_after_playback=True,
                            backend_used="kokoro",
                            is_cached=False,
                            normalized_text=norm_text
                        )

                logger.warning(f"[TTS] Síntesis con Kokoro falló para '{norm_text}'. Evaluando fallback...")

            # 5. Fallback a motor nativo (pyttsx3) o ejecución si motor es nativo
            if self.config.fallback_to_native or self.config.tts_engine.lower() != "kokoro":
                res = self._synthesize_native_temp(norm_text)
                if isinstance(res, tuple):
                    fallback_path, backend_name = res
                else:
                    fallback_path = res
                    backend_name = "pyttsx3_fallback"

                if fallback_path and os.path.isfile(fallback_path):
                    logger.info(f"[TTS] Fallback nativo ({backend_name}) generado exitosamente: {fallback_path}")
                    # El fallback NUNCA se persiste en el caché de Kokoro
                    return AudioArtifact(
                        wav_path=fallback_path,
                        cleanup_after_playback=True,
                        backend_used=backend_name,
                        is_cached=False,
                        normalized_text=norm_text
                    )

        logger.error(f"[TTS] Error crítico: Todos los backends de síntesis fallaron para '{logical_text}'")
        return None

    def _synthesize_native_temp(self, text: str) -> Tuple[Optional[str], str]:
        """
        Sintetiza texto a un archivo temporal mediante el fallback nativo del sistema operativo:
        - En macOS: /usr/bin/say con conversión WAV mediante afconvert.
        - En Windows: pyttsx3 con SAPI5.
        Retorna (temp_file_path, backend_name).
        """
        temp_dir = self.cache_manager.cache_dir if self.cache_manager.enabled else "data/audio_cache"
        os.makedirs(temp_dir, exist_ok=True)
        base_id = f"{os.getpid()}_{int(time.monotonic() * 1000)}"

        # 1. Fallback macOS (Apple Silicon M1 / macOS Intel)
        if sys.platform == "darwin":
            path = self._synthesize_macos_native(text, temp_dir, base_id)
            return path, "macos_say_fallback"

        # 2. Fallback Windows (SAPI5 / pyttsx3)
        path = self._synthesize_pyttsx3_native(text, temp_dir, base_id)
        return path, "pyttsx3_fallback"

    def _synthesize_macos_native(self, text: str, temp_dir: str, base_id: str) -> Optional[str]:
        """Sintetiza mediante /usr/bin/say nativo de Apple y convierte a WAV lineal PCM."""
        try:
            import subprocess
            temp_aiff = os.path.join(temp_dir, f".fallback_{base_id}.aiff")
            temp_wav = os.path.join(temp_dir, f".fallback_{base_id}.wav")

            # Invocación nativa say (soporta voces en español como Monica, Paulina, Jorge)
            cmd_say = ["say", "-o", temp_aiff, text]
            res_say = subprocess.run(cmd_say, capture_output=True, check=False)
            if res_say.returncode != 0:
                logger.warning(f"[TTSEngine] Fallo en /usr/bin/say: {res_say.stderr.decode('utf-8', errors='ignore')}")
                return None

            if not os.path.isfile(temp_aiff):
                return None

            # Conversión canónica a WAV RIFF 16-bit PCM mediante afconvert
            cmd_conv = ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", temp_aiff, temp_wav]
            res_conv = subprocess.run(cmd_conv, capture_output=True, check=False)

            # Limpiar el archivo AIFF intermedio
            try:
                os.remove(temp_aiff)
            except OSError:
                pass

            if res_conv.returncode == 0 and os.path.isfile(temp_wav) and os.path.getsize(temp_wav) > 44:
                return temp_wav
            elif os.path.isfile(temp_aiff) and os.path.getsize(temp_aiff) > 44:
                # Si afconvert no estaba disponible, afplay reproduce AIFF nativamente
                return temp_aiff

            return None
        except Exception as e:
            logger.error(f"[TTSEngine] Error en fallback macOS say: {e}")
            return None

    def _synthesize_pyttsx3_native(self, text: str, temp_dir: str, base_id: str) -> Optional[str]:
        """Sintetiza texto a un WAV temporal mediante pyttsx3 (Windows/SAPI5)."""
        try:
            temp_path = os.path.join(temp_dir, f".fallback_{base_id}.wav")
            eng = self._create_native_engine()
            if eng is None:
                return None
            eng.save_to_file(text, temp_path)
            eng.runAndWait()

            if os.path.isfile(temp_path) and os.path.getsize(temp_path) > 44:
                return temp_path
            return None
        except Exception as e:
            logger.error(f"[TTSEngine] Fallo en síntesis nativa pyttsx3: {e}")
            return None

    def _create_native_engine(self):
        """Crea una instancia fresca del motor nativo pyttsx3 si está disponible."""
        if pyttsx3 is None:
            logger.warning("[TTSEngine] pyttsx3 no está disponible en este entorno.")
            return None

        if sys.platform == "win32":
            try:
                import comtypes
                comtypes.CoInitialize()
            except Exception:
                pass

        try:
            eng = pyttsx3.init()
            eng.setProperty("rate", self.config.voice_rate)
            eng.setProperty("volume", self.config.voice_volume)

            voices = eng.getProperty("voices")
            for voice in voices:
                v_id = getattr(voice, "id", "").lower()
                v_name = getattr(voice, "name", "").lower()
                if "spanish" in v_name or "es-es" in v_id or "es_" in v_id or "helena" in v_id or "monica" in v_id:
                    eng.setProperty("voice", voice.id)
                    break
            return eng
        except Exception as e:
            logger.debug(f"[TTSEngine] Error configurando voz nativa: {e}")
            return None

    def synthesize_to_file(self, text: str, output_wav_path: str) -> bool:
        """
        Método de compatibilidad hacia atrás estricta.
        Sintetiza directamente al archivo destino solicitado.
        """
        if not self._is_initialized:
            self.initialize()

        norm_text = self.normalizer.normalize(text)
        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)

        # 1. Si Kokoro está disponible y listo, sintetizar con Kokoro
        if self.config.tts_engine.lower() == "kokoro":
            if self._kokoro_backend is None:
                self._kokoro_backend = KokoroBackend(
                    model_path=self.config.kokoro_model_path,
                    voices_path=self.config.kokoro_voices_path,
                    voice=self.config.kokoro_voice,
                    language=self.config.kokoro_language,
                    model_version=self.config.kokoro_model_version
                )
            ok = self._kokoro_backend.synthesize_to_file(norm_text, output_wav_path)
            if ok and os.path.isfile(output_wav_path) and os.path.getsize(output_wav_path) > 44:
                return True

        # 2. Fallback a motor nativo del SO
        if sys.platform == "darwin":
            import subprocess
            temp_aiff = output_wav_path + ".aiff"
            res_say = subprocess.run(["say", "-o", temp_aiff, norm_text], capture_output=True, check=False)
            if res_say.returncode == 0 and os.path.isfile(temp_aiff):
                res_conv = subprocess.run(
                    ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", temp_aiff, output_wav_path],
                    capture_output=True,
                    check=False
                )
                try:
                    os.remove(temp_aiff)
                except OSError:
                    pass
                if res_conv.returncode == 0 and os.path.isfile(output_wav_path) and os.path.getsize(output_wav_path) > 44:
                    return True

        # 3. Fallback a pyttsx3 (Windows/SAPI5)
        try:
            eng = self._create_native_engine()
            if eng:
                eng.save_to_file(norm_text, output_wav_path)
                eng.runAndWait()
                return os.path.isfile(output_wav_path) and os.path.getsize(output_wav_path) > 44
            return False
        except Exception as e:
            logger.error(f"[TTSEngine] Error en synthesize_to_file: {e}")
            return False
