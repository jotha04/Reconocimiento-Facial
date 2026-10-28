"""
FR2000 - Servidor Web de Alto Rendimiento (FastAPI + MJPEG Streaming)
Provee la interfaz gráfica web en tiempo real, streaming de video y telemetría continua.
"""

import os
import sys
import time
import logging
from typing import Optional
from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Asegurar importación de módulos raíz
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.pipeline_bridge import PipelineBridge

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("FR2000-Web")

# Instancia global de FastAPI
app = FastAPI(title="FR2000 Web Dashboard", version="1.0.0")

# Montar recursos estáticos
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Inicializar puente del pipeline
bridge: Optional[PipelineBridge] = None


@app.on_event("startup")
def startup_event():
    global bridge
    logger.info("[FR2000] Iniciando motor biométrico y puente de streaming...")
    bridge = PipelineBridge(config_path="config/audacia_config.yaml")
    bridge.start()
    logger.info("[FR2000] Servidor Web listo en http://localhost:8000")


@app.on_event("shutdown")
def shutdown_event():
    global bridge
    if bridge:
        logger.info("[FR2000] Deteniendo puente biométrico...")
        bridge.stop()


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Sirve la página web principal del dashboard FR2000."""
    index_path = os.path.join(templates_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Retorna el favicon ICO biométrico nativo."""
    ico_path = os.path.join(static_dir, "favicon.ico")
    if os.path.isfile(ico_path):
        with open(ico_path, "rb") as f:
            return Response(content=f.read(), media_type="image/x-icon")
    svg_icon = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2300f0ff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='8'/></svg>"""
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/api/ready")
def check_ready():
    """Retorna estado de preparación del pipeline y cámara."""
    if bridge is not None and bridge.get_jpeg_frame() is not None:
        return {"ready": True, "status": "streaming"}
    return JSONResponse(status_code=503, content={"ready": False, "status": "initializing"})


def generate_mjpeg_stream():
    """Generador eficiente de fotogramas MJPEG para el navegador web sin duplicados."""
    last_frame_id = -1
    while True:
        if bridge is None:
            time.sleep(0.05)
            continue
        
        frame_id, frame_bytes = bridge.wait_for_new_frame(last_frame_id, timeout=0.04)
        if frame_bytes is not None and frame_id > last_frame_id:
            last_frame_id = frame_id
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.get("/video_feed")
def video_feed():
    """Endpoint del flujo de video en vivo."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/telemetry")
def get_telemetry():
    """Retorna la telemetría biométrica, FPS, latencia e historial en tiempo real."""
    if bridge is None:
        return JSONResponse({"status": "initializing"})
    return JSONResponse(bridge.get_telemetry_payload())


class ViewModePayload(BaseModel):
    mode: str

@app.post("/api/controls/view_mode")
def set_view_mode(payload: ViewModePayload):
    """Alterna entre vista Combinada, Cámara y Profundidad."""
    if bridge:
        bridge.set_view_mode(payload.mode)
    return {"status": "ok", "mode": payload.mode}


@app.post("/api/controls/toggle_pause")
def toggle_pause():
    """Pausa o reanuda la captura de video."""
    if bridge:
        paused = bridge.toggle_pause()
        return {"status": "ok", "is_paused": paused}
    return {"status": "error"}


class ThresholdPayload(BaseModel):
    threshold: float

@app.post("/api/controls/threshold")
def set_threshold(payload: ThresholdPayload):
    """Calibra el umbral de detección de YuNet."""
    if bridge:
        bridge.set_score_threshold(payload.threshold)
    return {"status": "ok", "threshold": payload.threshold}


class SimThresholdPayload(BaseModel):
    threshold: float

@app.post("/api/controls/similarity_threshold")
def set_similarity_threshold(payload: SimThresholdPayload):
    """Calibra el umbral de similitud facial de SFace."""
    if bridge:
        bridge.set_similarity_threshold(payload.threshold)
    return {"status": "ok", "threshold": payload.threshold}


class AudioTogglePayload(BaseModel):
    enabled: bool

@app.post("/api/controls/toggle_audio")
def toggle_audio(payload: AudioTogglePayload):
    """Activa o silencia los saludos por voz."""
    if bridge:
        bridge.set_audio_enabled(payload.enabled)
    return {"status": "ok", "enabled": payload.enabled}


@app.post("/api/actions/reload_identities")
def reload_identities():
    """Recarga las identidades desde SQLite."""
    if bridge:
        count = bridge.reload_identities()
        return {"status": "ok", "embeddings_count": count}
    return {"status": "error"}


@app.post("/api/actions/snapshot")
def take_snapshot():
    """Guarda una captura diagnóstica en disco."""
    if bridge:
        filepath = bridge.save_snapshot()
        return {"status": "ok", "filepath": filepath}
    return {"status": "error"}


class RegisterCameraPayload(BaseModel):
    name: str

@app.post("/api/identities/register_camera")
def register_camera(payload: RegisterCameraPayload):
    """Registra una persona capturando el fotograma en vivo con normalización fotométrica."""
    if not bridge:
        return JSONResponse({"status": "error", "message": "Motor no inicializado."}, status_code=500)
    result = bridge.register_from_camera(payload.name)
    return JSONResponse(result, status_code=200 if result.get("success") else 400)


@app.post("/api/identities/register_upload")
async def register_upload(file: UploadFile = File(...), name: str = Form(...)):
    """Registra una persona a partir de una foto subida con normalización fotométrica."""
    if not bridge:
        return JSONResponse({"status": "error", "message": "Motor no inicializado."}, status_code=500)
    contents = await file.read()
    result = bridge.register_from_image_bytes(contents, name)
    return JSONResponse(result, status_code=200 if result.get("success") else 400)


@app.delete("/api/identities/{person_id}")
def delete_identity(person_id: str):
    """Elimina una identidad de la base de datos."""
    if not bridge:
        return JSONResponse({"status": "error", "message": "Motor no inicializado."}, status_code=500)
    result = bridge.delete_person(person_id)
    return JSONResponse(result, status_code=200 if result.get("success") else 400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
