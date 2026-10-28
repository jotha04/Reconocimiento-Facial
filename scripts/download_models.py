#!/usr/bin/env python3
import os
import sys
import urllib.request

url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "kokoro", "kokoro-v1.0.onnx"))

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rProgreso: {percent:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)")
        sys.stdout.flush()

def main():
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10 * 1024 * 1024:
        print("El modelo kokoro-v1.0.onnx ya esta descargado.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Descargando kokoro-v1.0.onnx en {output_path}...")
    urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
    print("\nDescarga completa.")

if __name__ == "__main__":
    main()
