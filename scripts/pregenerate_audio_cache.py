"""
Herramienta de Pre-generación Determinista de Caché de Audio (Fase 7).
Permite sintetizar por adelantado las frases recurrentes y saludos oficiales de AudacIA,
poblando 'data/audio_cache/' para que durante la exhibición pública las locuciones
sean recuperadas mediante Cache Hits instantáneos (< 0.2 ms).
"""

import os
import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audacia.app.attention.config import load_audacia_config
from audacia.app.audio.tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("AudacIA-Pregen")

# Frases oficiales deterministas para pre-generación inicial
DEFAULT_CORE_PHRASES = [
    "Bienvenido a AudacIA.",
    "Hola, bienvenidos a AudacIA.",
    "Hola, bienvenido a AudacIA, Jonathan.",
    "Hola, bienvenida a AudacIA, María.",
    "Hola, bienvenidos a AudacIA, Jonathan y María."
]


def pregenerate_cache(phrases=None, config_path="config/audacia_config.yaml"):
    print("=" * 70)
    print("    AUDACIA — PRE-GENERACIÓN DE CACHÉ DE AUDIO PERSISTENTE (FASE 7)   ")
    print("=" * 70)

    cfg_dict = load_audacia_config(config_path)
    audio_cfg = cfg_dict["audio"]

    if not audio_cfg.cache_enabled:
        print("[AVISO] El caché está deshabilitado en la configuración. Abortando pregeneración.")
        return

    engine = TTSEngine(audio_cfg)
    engine.initialize()

    target_phrases = phrases or DEFAULT_CORE_PHRASES
    print(f"\nProcesando {len(target_phrases)} frases con backend '{audio_cfg.tts_engine}' (Voz: '{audio_cfg.kokoro_voice}')...\n")

    generated_count = 0
    cached_count = 0

    for idx, phrase in enumerate(target_phrases, start=1):
        t0 = time.perf_counter()
        artifact = engine.get_audio_artifact(phrase)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not artifact:
            print(f"[{idx}/{len(target_phrases)}] ERROR al procesar: '{phrase}'")
            continue

        status = "CACHE HIT" if artifact.backend_used == "cache" else f"SYNTHESIZED ({artifact.backend_used})"
        if artifact.backend_used == "cache":
            cached_count += 1
        else:
            generated_count += 1

        print(f"[{idx}/{len(target_phrases)}] [{status:20}] ({elapsed_ms:6.1f} ms) '{phrase}'")
        print(f"       -> WAV: {artifact.wav_path}")
        print(f"       -> Texto Normalizado: '{artifact.normalized_text}'\n")

    print("=" * 70)
    print(f"Resumen: {cached_count} existentes en caché, {generated_count} nuevos sintetizados.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generador de Caché de Audio para AudacIA")
    parser.add_argument("--config", default="config/audacia_config.yaml", help="Ruta al archivo de configuración")
    parser.add_argument("--phrase", action="append", help="Frase específica a pre-generar (puede repetirse)")
    args = parser.parse_args()

    pregenerate_cache(phrases=args.phrase, config_path=args.config)
