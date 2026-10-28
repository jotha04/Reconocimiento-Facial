import os
from pathlib import Path

# Calcula la raíz del proyecto asumiendo que este archivo está en audacia/app/core/paths.py
# __file__ -> audacia/app/core/paths.py
# parent 1 -> core
# parent 2 -> app
# parent 3 -> audacia
# parent 4 -> PROJECT_ROOT (camara)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def resolve_path(relative_path: str) -> str:
    """Resuelve una ruta relativa contra el PROJECT_ROOT absoluto."""
    if not relative_path:
        return relative_path
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)
