#!/usr/bin/env python3
"""
FR2000 - Servidor de Panel Web en Vivo
Permite iniciar el visor web con: python run_web.py
"""
import os
import sys
import argparse
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="FR2000 Web Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Dirección IP de escucha (por defecto 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto HTTP (por defecto 8000)")
    args = parser.parse_args()

    print("=" * 70)
    print("      FR2000 — PANEL DE CONTROL & TELEMETRÍA BIOMÉTRICA EN VIVO       ")
    print(f"      Acceso web disponible en: http://localhost:{args.port}          ")
    print("=" * 70)

    from web.app import app
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
