#!/bin/bash
# ==============================================================================
# AudacIA / FR2000 - Script de Inicio del Servidor Web para macOS
# Inicia el dashboard en http://localhost:8000 y abre el navegador automáticamente.
# ==============================================================================

# 1. Asegurar que el directorio de trabajo sea la raíz del proyecto
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

echo "=================================================================="
echo "          AUDACIA FR2000 - DASHBOARD BIOMÉTRICO LOCAL            "
echo "=================================================================="
echo "Directorio de trabajo: $DIR"

# 2. Verificar existencia y activar entorno virtual
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activando entorno virtual (.venv)..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activando entorno virtual (venv)..."
    source "venv/bin/activate"
else
    echo ""
    echo "------------------------------------------------------------------"
    echo " [ERROR] No se detectó un entorno virtual configurado (.venv)."
    echo " Abre una terminal y ejecuta:"
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo "------------------------------------------------------------------"
    read -p "Presiona Enter para salir..."
    exit 1
fi

PORT=8000
URL="http://localhost:$PORT"
READY_URL="http://127.0.0.1:$PORT/api/ready"

# 3. Esperar activamente a que el servidor HTTP y la cámara estén 100% listos antes de abrir el navegador
(
    for i in {1..60}; do
        if curl -s -m 1 "$READY_URL" 2>/dev/null | grep -q '"ready":true'; then
            sleep 0.3
            open "$URL"
            exit 0
        fi
        sleep 0.5
    done
    open "$URL"
) &

# 4. Iniciar servidor web FastAPI / Uvicorn
echo "[INFO] Iniciando servidor web en $URL..."
echo "[INFO] Presiona Ctrl+C en esta ventana para detener el servidor."
echo "=================================================================="
echo ""

python run_web.py --port "$PORT"

EXIT_CODE=$?
echo ""
echo "------------------------------------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Servidor detenido correctamente."
else
    echo "Servidor finalizado con código: $EXIT_CODE"
fi
read -p "Presiona [Enter] para cerrar esta ventana..."
