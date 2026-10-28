#!/usr/bin/env python3
"""
Script de descarga automática para modelos neuronales pesados (>100MB)
que no se almacenan directamente en GitHub por límites de cuota.
"""

import os
import sys
import urllib.request

MODELS = [
    {
        "name": "Kokoro-v1.0 ONNX (TTS Neuronal)",
        "path": os.path.join("models", "kokoro", "kokoro-v1.0.onnx"),
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "expected_size_mb": 310
    }
]

def download_with_progress(url: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progreso: {percent:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)")
            sys.stdout.flush()

    print(f"Descargando desde:\n  {url}")
    print(f"Destino:\n  {output_path}")
    urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
    print("\n  Descarga finalizada con éxito.")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(base_dir)
    print("=" * 60)
    print(" AudacIA - Verificación y Descarga de Modelos Neuronales")
    print("=" * 60)

    for item in MODELS:
        target = os.path.join(base_dir, item["path"])
        if os.path.exists(target) and os.path.getsize(target) > 10 * 1024 * 1024:
            size_mb = os.path.getsize(target) / (1024 * 1024)
            print(f"[OK] {item['name']} ya existe ({size_mb:.1f} MB). Omitiendo.")
        else:
            print(f"[FALTA] {item['name']} no encontrado.")
            try:
                download_with_progress(item["url"], target)
            except Exception as e:
                print(f"\n[ERROR] Falló la descarga: {e}")
                print(f"Puedes descargarlo manualmente desde: {item['url']}")
                print(f"y ubicarlo en: {item['path']}")

    print("\nVerificación de modelos completada.")

if __name__ == "__main__":
    main()

