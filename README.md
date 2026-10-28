# AudacIA — Sistema de Interacción Social y Percepción Autónoma para Robots

AudacIA es una plataforma integral de visión computacional, atención visual, biometría facial y síntesis de voz natural offline diseñada para operar en un **robot de exhibición y recepción autónomo**. 

El sistema está optimizado para ejecutarse localmente y en tiempo real (**30 FPS sostenidos**) en un **Mac mini M1 (Apple Silicon ARM64, macOS)** utilizando una cámara web externa **Logitech C270 HD**, sin depender de internet ni de servicios en la nube.

---

## 🎯 ¿Para qué sirve AudacIA?

AudacIA transforma una cámara y un altavoz en un anfitrión robótico inteligente:
1. **Detección y Tracking Visual:** Detecta rostros y calcula la orientación de la cabeza en tiempo real mediante redes neuronales ultraligeras (**YuNet** en CPU con <8 ms de latencia).
2. **Motor de Atención Conjuntiva:** Filtra transeúntes irrelevantes o personas que caminan de espaldas/perfil, identificando únicamente a usuarios con **intención real de interactuar** frente al robot.
3. **Reconocimiento Facial Biométrico Offline:** Identifica usuarios conocidos mediante **SFace** (128-D embeddings y SQLite local), saludándolos por su nombre de forma personalizada.
4. **Resolución de Dinámicas Sociales y Grupos:** Distingue entre individuos solitarios, parejas o grupos mixtos (conocidos + desconocidos), emitiendo saludos plurales coherentes (*"Hola, bienvenidos a AudacIA..."*) y evitando saludos redundantes o repetitivos gracias a memoria de interacción y cooldowns espaciales.
5. **Síntesis de Voz Humana Offline (Kokoro-82M):** Locución de alta calidad en español mediante el modelo neuronal Kokoro con aceleración ONNX, gestión de caché instantánea (<1.5 ms para saludos recurrentes) y normalización fonética personalizada.

---

## 📁 Estructura del Paquete de Producción

```text
camara/
├── run_audacia.command             # 🚀 Lanzador de doble clic para macOS
├── main.py                         # Punto de entrada alternativo en Python
├── requirements.txt                # Dependencias multiplataforma (macOS / Windows)
├── README.md                       # Manual de instalación y operación
├── config/
│   └── audacia_config.yaml         # Configuración central (umbrales, audios, cooldowns)
├── audacia/                        # Código fuente del sistema
│   └── app/
│       ├── camera/                 # Descubrimiento determinista de C270 (AVFoundation)
│       ├── detection/              # Detector facial YuNet y orientación geométrica
│       ├── tracking/               # Rastreo multi-rostro con Filtro de Kalman e IoU
│       ├── attention/              # Motor de atención y estados (OBSERVED -> ENGAGING)
│       ├── recognition/            # Biometría SFace y base de datos de identidades
│       ├── interaction/            # Formación de grupos y control de saludos
│       ├── audio/                  # Pipeline TTS Kokoro, caché WAV y normalización
│       └── diagnostics/            # Visualizador HUD y telemetría en tiempo real
├── models/
│   ├── face_detection_yunet_2023mar.onnx     # Modelo de visión YuNet
│   ├── face_recognition_sface_2021dec.onnx   # Modelo biométrico SFace
│   └── kokoro/
│       ├── kokoro-v1.0.onnx                  # Modelo neuronal de voz Kokoro-82M
│       └── voices-v1.0.bin                   # Banco de voces en español (ef_dora)
├── data/
│   ├── identities.db               # Base de datos SQLite con usuarios registrados
│   └── audio_cache/                # Caché persistente de saludos pre-sintetizados
└── scripts/
    ├── run_diagnostics.py          # Script principal del visor interactivo
    ├── register_person.py          # Herramienta para registrar nuevos rostros
    ├── pregenerate_audio_cache.py  # Herramienta para calentar la caché de audio
    └── benchmark_runner.py         # Herramienta para medir FPS y latencias en hardware
```

---

## 💻 Requisitos de Hardware y Software

* **Hardware:**
  * Mac mini M1 / MacBook (Apple Silicon M1/M2/M3) o Mac Intel con macOS 12+ (Monterey o superior).
  * Cámara web externa **Logitech C270 HD WebCam** conectada por USB.
  * Altavoces o salida de audio del Mac activada.
* **Software:**
  * **Python 3.10 o 3.11** nativo (ARM64 para Apple Silicon).
  * **Homebrew** instalado en macOS ([brew.sh](https://brew.sh)).

---

## 🛠️ Instalación Paso a Paso en Mac

Abre la aplicación **Terminal** en macOS y ejecuta los siguientes pasos:

### Paso 1: Instalar la dependencia de sistema para fonemas
Kokoro TTS requiere la librería de fonemización `espeak-ng`. Instálala con Homebrew:
```bash
brew install espeak-ng
```

### Paso 2: Crear el entorno virtual de Python
Navega a la carpeta donde descomprimiste el proyecto:
```bash
cd ruta/hacia/camara
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar las librerías de Python
Actualiza `pip` e instala las dependencias del proyecto:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **Nota:** `requirements.txt` ya incluye selectores de plataforma automáticos (`sys_platform`), por lo que en Mac omitirá dependencias innecesarias de Windows (como pygrabber o comtypes) e instalará exactamente los paquetes necesarios.

### Paso 4: Descargar el modelo de voz Kokoro TTS (>100MB)
Por límites de cuota de almacenamiento en GitHub, el modelo neuronal de voz `kokoro-v1.0.onnx` se descarga automáticamente ejecutando:
```bash
python scripts/download_models.py
```

### Paso 5: Dar permisos de ejecución a los lanzadores `.command`
Para poder abrir el programa o la web haciendo doble clic desde el Finder:
```bash
chmod +x run_web.command
```

### Paso 5: Conceder permisos de Cámara en macOS (Paso Crítico de Privacidad)
macOS restringe el acceso a la cámara por políticas de seguridad:
1. Conecta la cámara Logitech C270 por USB al Mac.
2. Ejecuta una prueba rápida en la Terminal:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION); ret, f = cap.read(); cap.release()"
   ```
3. macOS mostrará una ventana emergente: **"Terminal desea acceder a la cámara"**. Haz clic en **Permitir**.
   *(Si por error hiciste clic en denegar, ve a: Preferencias del Sistema / Ajustes del Sistema $\to$ Privacidad y Seguridad $\to$ Cámara y activa la casilla para Terminal).*

---

## 🚀 Cómo Iniciar el Programa

Tienes dos formas de ejecutar AudacIA:

### Opción A (Recomendada): Doble Clic en el Finder
1. Abre la carpeta del proyecto en el **Finder**.
2. Haz **doble clic sobre el archivo `run_audacia.command`**.
3. Se abrirá automáticamente una ventana de Terminal, activará el entorno virtual y desplegará la interfaz gráfica con el video de la cámara y el HUD de diagnóstico.

### Opción B: Desde la Terminal
```bash
source .venv/bin/activate
python main.py
```
*(O también: `python scripts/run_diagnostics.py`)*

---

## 🎨 Interpretación de Colores y Estados en Pantalla (HUD)

En la ventana de video observarás recuadros alrededor de los rostros detectados con distintos colores según su estado de interacción:

| Color de Caja | Estado | Significado |
| :---: | :---: | :--- |
| **Azul** | `OBSERVED` | Rostro recién detectado en el encuadre, pero aún no demuestra atención sostenida o está mirando hacia otro lado. |
| **Amarillo** | `ATTENTION_CANDIDATE` | La persona lleva más de 0.4s presente y orientada razonablemente hacia el robot. Es candidata a interactuar o a ser acompañante en un grupo. |
| **Naranja** | `ENGAGING` | La persona lleva más de 1.2s mirando fijamente al robot de frente. **Dispara la interacción activa, biometría SFace y saludo de voz.** |
| **Gris** | `DISENGAGED` | Persona que estuvo atenta pero apartó la mirada o se alejó. Si permanece físicamente estable al lado de alguien en `ENGAGING`, el sistema la incluye en el saludo grupal (*"Hola, bienvenidos..."*). |

---

## 🎮 Controles en Pantalla (Ventana de Video)

Con la ventana gráfica de video seleccionada, puedes usar los siguientes atajos de teclado:

| Tecla | Acción |
| :---: | :--- |
| **`q`** o **`ESC`** | **Salir** del programa de forma ordenada cerrando cámara y audio. |
| **`b`** | Imprime un **snapshot de rendimiento y telemetría** en la consola (FPS, latencia YuNet, inferencia SFace, duración TTS). |
| **`s`** | Guarda una captura instantánea del frame con el HUD en `data/snapshots/`. |
| **`+`** / **`-`** | Aumenta o disminuye en $\pm 0.05$ el umbral de detección de YuNet. |

---

## 👤 Cómo Registrar Nuevas Personas (Biometría Facial)

Para que el robot reconozca a alguien por su nombre y lo salude de forma personalizada:

1. Con la cámara conectada y el sujeto sentado de frente con buena iluminación, ejecuta:
   ```bash
   python scripts/register_person.py --name "Nombre_De_La_Persona"
   ```
2. El sistema capturará muestras faciales válidas (de frente, sin desenfoque) y almacenará su vector biométrico 128-D en `data/identities.db`.
3. Para listar los usuarios ya registrados:
   ```bash
   python scripts/register_person.py --list
   ```
4. Para eliminar un registro:
   ```bash
   python scripts/register_person.py --delete "Nombre_De_La_Persona"
   ```

---

## 🔊 Calentamiento y Pre-generación de Audios

AudacIA genera y guarda en disco los audios de los saludos más comunes para que suenen de forma **instantánea** (<1.5 ms) sin esperar al sintetizador neuronal.

Si registraste una persona nueva o cambiaste los textos de saludo en la configuración, puedes pre-generar los audios en la caché ejecutando:
```bash
python scripts/pregenerate_audio_cache.py
```

---

## ⚙️ Configuración y Personalización

El archivo [`config/audacia_config.yaml`](file:///config/audacia_config.yaml) permite ajustar todo el comportamiento del robot:

* **Textos de los saludos:**
  ```yaml
  greetings:
    single_known: "Hola, bienvenido a AudacIA, {name}."
    multiple_known: "Hola, bienvenidos a AudacIA, {names}."
    single_unknown: "Bienvenido a AudacIA."
    multiple_unknown: "Hola, bienvenidos a AudacIA."
    mixed_group: "Hola, bienvenidos a AudacIA."
  ```
* **Pronunciación y Fonética:** Si algún nombre suena extraño o con acento extranjero, puedes forzar su pronunciación en español:
  ```yaml
  audio:
    pronunciation_overrides:
      AudacIA: "Audacia"
      Jonathan: "Yónatan"
      Joshua: "Yóshua"
  ```
* **Cooldowns:** Tiempo que debe pasar antes de volver a saludar a la misma persona:
  ```yaml
  interaction:
    cooldown_known_seconds: 30.0    # 30 segundos para conocidos
    cooldown_unknown_seconds: 15.0  # 15 segundos para desconocidos
  ```

---

## 📊 Medición de Rendimiento en el Mac (Benchmark)

Para verificar que el Mac mini M1 esté operando al máximo rendimiento sin cuello de botella:
```bash
python scripts/benchmark_runner.py
```
El script medirá el framerate efectivo (debe dar **28 a 30 FPS**) y reportará la latencia media por componente (YuNet en CPU M1 debe tomar menos de 8 ms por cuadro).

---

## 🔍 Solución de Problemas Frecuentes

1. **Error: "NO se encontró la cámara Logitech C270 externa conectada"**
   - Asegúrate de que el cable USB esté firmemente conectado al Mac.
   - En Mac, la C270 es detectada bajo el backend AVFoundation. Desconecta y vuelve a conectar la cámara.
2. **Error: "libespeak-ng.dylib not found"**
   - Falta la dependencia nativa de fonemas. Ejecuta en Terminal: `brew install espeak-ng`.
   - Si persiste en Mac Apple Silicon, exporta la ruta en la terminal:
     ```bash
     export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
     ```
3. **La cámara se ve negra o no abre la ventana:**
   - Verifica los permisos en *Ajustes del Sistema $\to$ Privacidad y Seguridad $\to$ Cámara* y asegúrate de que Terminal tiene permiso activado.
4. **La persona está presente pero el recuadro se queda en azul (`OBSERVED`):**
   - Verifica la iluminación ambiental. Si hay sombras muy marcadas sobre el rostro, acércate o mira directamente al lente de la cámara C270 durante ~1.2 segundos para alcanzar el estado de interacción (`ENGAGING`).
5. **No se escucha el audio:**
   - En macOS, AudacIA usa la herramienta nativa `afplay`. Comprueba el volumen del Mac y la salida de audio seleccionada en el menú de sonido (altavoces integrados, USB o jack 3.5mm).
