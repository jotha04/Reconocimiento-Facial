#!/usr/bin/env python3
"""
AudacIA - Punto de Entrada Principal (Producción)
Permite iniciar la aplicación con: python main.py
"""
import sys
import os

# Asegurar que el directorio raíz esté en sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    if "--web" in sys.argv:
        from run_web import main as web_main
        web_main()
    else:
        from scripts.run_diagnostics import main
        main()
