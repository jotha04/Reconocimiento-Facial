# AudacIA — Sistema de Percepción Autónoma, Biometría Facial e Interacción Social

Plataforma integral de visión por computador, reconocimiento biométrico facial 100% offline, análisis de atención visual y síntesis de voz natural neuronal diseñada para operar en un **robot autónomo anfitrión y recepcionista**.

El sistema procesa video a **30 FPS continuos** sin depender de internet ni de APIs externas, garantizando privacidad absoluta de datos y latencias inferiores a 10 ms en inferencia visual.

---

## 🎯 ¿Para qué sirve AudacIA?

AudacIA dota a una cámara web y un sistema de sonido de las capacidades cognitivas de un anfitrión social robótico:

1. **Detección y Orientación Facial en Tiempo Real:** Detecta múltiples rostros y estima la pose tridimensional de la cabeza mediante la red neuronal ultraligera **YuNet** (latencia <8 ms en CPU).
2. **Filtro de Intención y Atención Sostenida:** Descarta transeúntes ocasionales o personas de espaldas, detectando con precisión quién tiene **intención genuina de interactuar** frente al robot.
3. **Reconocimiento Facial Biométrico Offline:** Identifica usuarios conocidos mediante descriptores **SFace** (vectores biométricos de 128 dimensiones) y almacenamiento local en SQLite, saludándolos por su nombre.
4. **Resolución de Dinámicas Sociales y Grupos:** Diferencia entre visitantes individuales, parejas o grupos mixtos (conocidos + desconocidos), emitiendo saludos plurales coherentes y gestionando enfriamientos espaciales (*cooldowns*) para no repetir saludos.
5. **Síntesis de Voz Humana Offline (Kokoro-82M):** Locución de calidad humana en español mediante el modelo neuronal Kokoro (acelerado por ONNX), con caché instantánea (<1.5 ms) y normalización fonética de nombres propios.
6. **Doble Modo de Operación:** Visor HUD diagnóstico con ventana nativa OpenCV o Dashboard Web interactivo en tiempo real con streaming MJPEG.

---

## 📁 Estructura del Repositorio

```text
reconocimiento_facial/
├── main.py                         # Punto de entrada universal (CLI / HUD / Web)
├── run_web.py                      # Lanzador directo del servidor web FastAPI
├── run_web.command                 # 🚀 Lanzador de doble clic para macOS
├── requirements.txt                # Dependencias multiplataforma (macOS / Windows / Linux)
├── README.md                       # Documentación del sistema
├── config/
│   └── audacia_config.yaml         # Configuración central (umbrales, audios, cooldowns)
├── audacia/                        # Paquete principal del motor
│   └── app/
│       ├── camera/                 # Descubrimiento determinista de hardware (AVFoundation / DirectShow)
│       ├── detection/              # Detector facial YuNet y estimación de orientación
│       ├── tracking/               # Seguimiento multi-rostro con Filtro de Kalman e IoU
│       ├── attention/              # Máquina de estados de atención (OBSERVED -> ENGAGING)
│       ├── recognition/            # Biometría SFace y catálogo SQLite local
│       ├── interaction/            # Resolución de dinámicas sociales y generador de saludos
│       ├── audio/                  # Pipeline TTS Kokoro, caché WAV y normalización
│       └── diagnostics/            # Visualizador HUD y telemetría de rendimiento
├── models/
│   ├── face_detection_yunet_2023mar.onnx     # Modelo de detección YuNet
│   ├── face_recognition_sface_2021dec.onnx   # Modelo de embedding SFace
│   └── kokoro/
│       ├── voices-v1.0.bin                   # Banco de voces en español (ef_dora)
│       └── kokoro-v1.0.onnx                  # Modelo TTS (>100MB, se descarga con script)
├── data/                           # Directorio para base de datos y cachés locales
│   ├── audio_cache/                # Caché de archivos de audio sintetizados
│   └── snapshots/                  # Capturas de diagnóstico
├── uploads/                        # Directorio temporal para registro de fotos web
├── scripts/
│   ├── download_models.py          # Script de descarga para modelos pesados (>100MB)
│   ├── run_diagnostics.py          # Visor interactivo nativo
│   ├── register_person.py          # Registro biométrico por consola/cámara
│   ├── pregenerate_audio_cache.py  # Pre-renderizado de saludos recurrentes
│   └── benchmark_runner.py         # Medición de FPS y latencias de inferencia
└── web/                            # Frontend y Dashboard Web en tiempo real
    ├── app.py                      # Rutas FastAPI y endpoints biométricos
    ├── pipeline_bridge.py          # Puente asíncrono con el hilo de visión
    ├── static/                     # Estilos, scripts y assets del dashboard
    └── templates/                  # Plantilla HTML del dashboard
```

---

## 💻 Requisitos del Sistema

* **Sistemas Operativos:** macOS 12+ (optimizado para Apple Silicon M1/M2/M3), Windows 10/11 (x64), o Linux (Ubuntu 20.04+).
* **Hardware:**
  * Cámara web externa USB (e.g., Logitech C270 HD) o cámara integrada.
  * Parlante o salida de audio habilitada.
* **Software:**
  * Python **3.10 o 3.11**.
  * En macOS: `brew install espeak-ng` (necesario para el fonemizador de Kokoro).

---

## 🛠️ Instalación Paso a Paso

### 1. Clonar el Repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd reconocimiento_facial
```

### 2. Configurar el Entorno Virtual
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Descargar el Modelo Neuronal de Voz Kokoro (>100 MB)
Debido a las cuotas de almacenamiento de GitHub para archivos individuales mayores a 100 MB, el modelo `kokoro-v1.0.onnx` se descarga automáticamente ejecutando:
```bash
python scripts/download_models.py
```

---

## 🚀 Cómo Correr el Programa

AudacIA ofrece dos formas de ejecución según tus necesidades:

### Opción 1: Panel de Control Web en Vivo (Recomendado)
Proporciona una interfaz web moderna accesible desde cualquier navegador para monitorear el video en tiempo real, ver la telemetría y registrar nuevos usuarios fácilmente.

* **En macOS (Doble Clic):**
  Haz doble clic en `run_web.command`.
* **Desde la Terminal (macOS / Windows / Linux):**
  ```bash
  python run_web.py
  # o también:
  python main.py --web
  ```
* Abre tu navegador en: **`http://localhost:8000`**

### Opción 2: Visor Gráfico Nativo HUD (OpenCV)
Muestra una ventana directa con la cámara, recuadros de tracking, orientación facial y telemetría de rendimiento en pantalla:
```bash
python main.py
# o también:
python scripts/run_diagnostics.py
```

#### Atajos de Teclado en el Visor HUD:
| Tecla | Acción |
| :---: | :--- |
| **`q`** o **`ESC`** | Cierra la aplicación de forma ordenada. |
| **`b`** | Imprime un resumen de rendimiento y latencias en consola. |
| **`s`** | Guarda una captura instantánea en `data/snapshots/`. |
| **`+`** / **`-`** | Incrementa o reduce el umbral de detección de rostros. |

---

## 👤 Gestión de Identidades (Biometría Facial)

Para que el sistema reconozca a un nuevo usuario y lo salude por su nombre:

1. **Desde la Web:** Usa la sección de registro en el Dashboard Web (`http://localhost:8000`).
2. **Desde la Terminal:**
   ```bash
   # Registrar una persona usando la cámara:
   python scripts/register_person.py --name "Nombre_De_La_Persona"

   # Listar usuarios registrados:
   python scripts/register_person.py --list

   # Eliminar un usuario:
   python scripts/register_person.py --delete "Nombre_De_La_Persona"
   ```

---

## 🔊 Precalentamiento de Audios (Caché TTS)

Para garantizar saludos inmediatos (<1.5 ms de respuesta) al interactuar con el robot:
```bash
python scripts/pregenerate_audio_cache.py
```
Los saludos configurados se pre-sintetizarán en `data/audio_cache/`.

---

## ⚙️ Configuración y Personalización

Puedes modificar el comportamiento general en [`config/audacia_config.yaml`](file:///config/audacia_config.yaml):
* **Textos de bienvenida:** Personaliza los saludos para personas conocidas, grupos y desconocidos.
* **Tiempos de Cooldown:** Define el tiempo que debe transcurrir antes de volver a saludar a la misma persona.
* **Fonética:** Ajusta la pronunciación de nombres propios si el sintetizador no los acentúa de forma óptima.
