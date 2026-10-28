[Español](#español) | [English](#english)

---

<div id="español"></div>

# Reconocimiento Facial

Sistema de vision por computador, reconocimiento biometrico y sintesis de voz offline para interaccion en tiempo real. Utiliza YuNet para deteccion facial, SFace para reconocimiento biometrico (embeddings de 128 dimensiones) y Kokoro-82M (ONNX) para sintesis de voz local en español.

## Requisitos

- Python 3.10 o 3.11
- Camara web (USB o integrada)
- macOS / Linux / Windows
- `espeak-ng` (en macOS: `brew install espeak-ng`, requerido para el motor fonetico)

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

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Descargar el modelo de voz Kokoro:
El modelo `kokoro-v1.0.onnx` no esta incluido en el repositorio por temas de peso. Para descargarlo:
```bash
python scripts/download_models.py
```

## Uso

### Panel Web (FastAPI)
Para iniciar el servidor con streaming de video y panel de administracion web:
```bash
python run_web.py
```
O ejecutando `run_web.command` en macOS.
Disponible en: `http://localhost:8000`

### Modo Visor Directo (OpenCV)
Para correr el sistema directamente en ventana con metricas y recuadros de tracking:
```bash
python main.py
```

Controles de teclado:
- `q` / `ESC`: Salir
- `s`: Guardar captura del frame actual
- `b`: Imprimir telemetria en consola
- `+` / `-`: Subir o bajar umbral de deteccion

### Registro de Personas
Para enrolar un nuevo rostro en la base de datos local usando la camara:
```bash
python scripts/register_person.py --name "Nombre"
```
Para listar las personas registradas:
```bash
python scripts/register_person.py --list
```

## Configuracion

Parametros de tracking, saludos y tiempos de espera disponibles en `config/audacia_config.yaml`.

## Licencia

Distribuido bajo la licencia MIT. Consulta el archivo `LICENSE` para mas detalles.

---

<div id="english"></div>

# Facial Recognition

Computer vision system for face detection, tracking, biometric recognition, and offline text-to-speech synthesis for real-time human-robot interaction. Powered by YuNet for face detection, SFace for biometric recognition (128-dimensional embeddings), and Kokoro-82M (ONNX) for local Spanish voice synthesis.

## Requirements

- Python 3.10 or 3.11
- USB or integrated webcam
- macOS / Linux / Windows
- `espeak-ng` (on macOS: `brew install espeak-ng`, required by the phonemizer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jotha04/Reconocimiento-Facial.git
cd Reconocimiento-Facial
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Kokoro voice model:
The model file `kokoro-v1.0.onnx` is not tracked in the repository due to file size limits. Download it by running:
```bash
python scripts/download_models.py
```

## Usage

### Web Dashboard (FastAPI)
Starts the video streaming server and web dashboard:
```bash
python run_web.py
```
Or double-click `run_web.command` on macOS.
Access URL: `http://localhost:8000`

### Direct HUD Window (OpenCV)
Runs the standalone desktop interface with real-time metrics:
```bash
python main.py
```

Keyboard shortcuts:
- `q` / `ESC`: Quit
- `s`: Save frame snapshot
- `b`: Print telemetry benchmarks to console
- `+` / `-`: Increase or decrease detection threshold

### Enrolling New Faces
To enroll a new identity using the webcam:
```bash
python scripts/register_person.py --name "Name"
```
To list all registered identities:
```bash
python scripts/register_person.py --list
```

## Configuration

Detection thresholds, custom greeting phrases, and interaction cooldowns can be customized in `config/audacia_config.yaml`.

## License

Distributed under the MIT License. See the `LICENSE` file for details.
