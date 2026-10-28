# Reconocimiento Facial

Sistema de vision por computador, reconocimiento biometrico y sintesis de voz offline para interaccion en tiempo real. Utiliza YuNet para deteccion facial, SFace para reconocimiento biometrico (embeddings de 128 dimensiones) y Kokoro-82M (ONNX) para respuesta por voz en español.

## Requisitos

- Python 3.10 o 3.11
- Camara web (USB o integrada)
- macOS / Linux / Windows
- `espeak-ng` (en macOS: `brew install espeak-ng`, necesario para el motor fonetico de audio)

## Instalacion

1. Clonar el repositorio:
```bash
git clone https://github.com/jotha04/Reconocimiento-Facial.git
cd Reconocimiento-Facial
```

2. Crear y activar entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. Instalar librerias:
```bash
pip install -r requirements.txt
```

4. Descargar el modelo de voz Kokoro:
El archivo `kokoro-v1.0.onnx` no viene dentro del repositorio por limite de peso. Se descarga ejecutando:
```bash
python scripts/download_models.py
```

## Uso

### Panel Web (FastAPI)
Inicia la transmision de video y el panel de control en el navegador:
```bash
python run_web.py
```
O con doble clic en `run_web.command` en macOS.
Acceso: `http://localhost:8000`

### Modo Visor Directo (OpenCV)
Inicia la ventana con seguimiento y diagnostico en pantalla:
```bash
python main.py
```

Controles:
- `q` / `ESC`: Salir
- `s`: Guardar captura
- `b`: Ver telemetria en consola
- `+` / `-`: Modificar umbral de deteccion

### Registro de Usuarios
Para registrar nuevos rostros usando la camara:
```bash
python scripts/register_person.py --name "Nombre"
```
Para ver la lista de registrados:
```bash
python scripts/register_person.py --list
```

## Configuracion

Parametros de deteccion, saludos y tiempos de espera configurables en `config/audacia_config.yaml`.
