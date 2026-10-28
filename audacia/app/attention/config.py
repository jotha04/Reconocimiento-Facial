"""
Módulo de configuración tipada para Tracking, Attention Engine y Face Recognition.
Permite cargar desde archivo YAML o utilizar valores por defecto calibrados.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from audacia.app.core.paths import resolve_path


@dataclass
class CameraConfig:
    target_width: int = 1920
    target_height: int = 1080
    target_fps: int = 30
    use_threaded_reader: bool = True
    max_inference_dimension: Optional[int] = 1280


@dataclass
class DetectionConfig:
    model_path: str = "models/face_detection_yunet_2023mar.onnx"
    score_threshold: float = 0.65
    nms_threshold: float = 0.30
    min_face_size: int = 55


@dataclass
class TrackingConfig:
    max_age: int = 15              # Frames consecutivos tolerados en pérdida (~0.5s a 30 FPS)
    min_hits: int = 3              # Detecciones consecutivas para confirmar track (~0.1s)
    iou_threshold: float = 0.25    # Umbral mínimo de solapamiento IoU
    distance_threshold: float = 0.20 # Distancia máxima normalizada de centroides


@dataclass
class AttentionConfig:
    # Tiempos de persistencia (segundos)
    dwell_candidate_seconds: float = 0.50
    dwell_engaging_seconds: float = 1.20
    disengage_grace_seconds: float = 2.00

    # Umbrales de score
    candidate_score_threshold: float = 0.30
    engaging_score_threshold: float = 0.60
    min_orientation_for_engaging: float = 0.60
    frontal_orientation_grace_seconds: float = 0.70  # Tolerancia temporal a caídas transitorias de orientación frontal

    # Parámetros geométricos de rostro (píxeles en 640x480)
    min_face_height: int = 70
    optimal_face_height_min: int = 100
    optimal_face_height_max: int = 260
    max_face_height: int = 380

    # Cono central de atención
    central_zone_min_x: float = 0.20
    central_zone_max_x: float = 0.80

    # Tolerancia geométrica de orientación
    orientation_tolerance_sigma: float = 0.35


@dataclass
class RecognitionConfig:
    similarity_threshold: float = 0.55  # Umbral de similitud coseno (configurable)
    ambiguity_margin: float = 0.08      # Margen mínimo frente al segundo mejor candidato
    samples_for_consolidation: int = 3  # Muestras consistentes requeridas para confirmar KNOWN
    model_path: str = "models/face_recognition_sface_2021dec.onnx"
    db_path: str = "data/identities.db"
    adaptive_learning_enabled: bool = True  # Auto-adaptación pasiva para cambios graduales (barba/tiempo)
    adaptive_cooldown_seconds: float = 1800.0 # Intervalo mínimo entre auto-actualizaciones por persona

    def __post_init__(self):
        self.model_path = resolve_path(self.model_path)
        self.db_path = resolve_path(self.db_path)


@dataclass
class InteractionConfig:
    group_stabilization_seconds: float = 0.80   # Ventana breve para incorporar participantes simultáneos
    group_formation_max_seconds: float = 1.50   # Ventana máx si hay acompañantes prometedores en ATTENTION_CANDIDATE
    identity_resolution_timeout_seconds: float = 2.0 # Tiempo máx esperando reconocimiento
    session_timeout_seconds: float = 2.5        # Tiempo sin rostros en ENGAGING antes de cerrar sesión
    person_cooldown_seconds: float = 45.0       # Cooldown por identidad antes de poder repetir saludo
    unknown_cooldown_seconds: float = 25.0      # Cooldown para desconocidos
    allow_late_joiner_greeting: bool = False   # No interrumpir sesión activa


@dataclass
class GreetingsConfig:
    single_known: str = "Hola, bienvenido a AudacIA, {name}."
    multiple_known: str = "Hola, bienvenidos a AudacIA, {names}."
    single_unknown: str = "Bienvenido a AudacIA."
    multiple_unknown: str = "Hola, bienvenidos a AudacIA."
    mixed_group: str = "Hola, bienvenidos a AudacIA."


@dataclass
class AudioConfig:
    enabled: bool = True
    tts_engine: str = "kokoro"
    kokoro_voice: str = "ef_dora"
    kokoro_language: str = "es"
    kokoro_model_path: str = "models/kokoro/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "models/kokoro/voices-v1.0.bin"
    kokoro_model_version: str = "kokoro-v1.0"
    cache_enabled: bool = True
    cache_dir: str = "data/audio_cache"
    lazy_load: bool = True
    fallback_to_native: bool = True
    voice_rate: int = 155
    voice_volume: float = 1.0
    voice_language_preference: str = "es"
    pronunciation_overrides: Dict[str, str] = field(default_factory=lambda: {
        "AudacIA": "Audacia",
        "Jonathan": "Yónatan",
        "Joshua": "Yóshua",
    })

    def __post_init__(self):
        p_model = resolve_path(self.kokoro_model_path)
        if not os.path.isfile(p_model):
            alt_m1 = resolve_path("models/kokoro/kokoro-v1.0.onnx")
            alt_m2 = resolve_path("scratch/models/kokoro/kokoro-v1.0.onnx")
            if os.path.isfile(alt_m1):
                p_model = alt_m1
            elif os.path.isfile(alt_m2):
                p_model = alt_m2
        self.kokoro_model_path = p_model

        p_voices = resolve_path(self.kokoro_voices_path)
        if not os.path.isfile(p_voices):
            alt_v1 = resolve_path("models/kokoro/voices-v1.0.bin")
            alt_v2 = resolve_path("scratch/models/kokoro/voices-v1.0.bin")
            if os.path.isfile(alt_v1):
                p_voices = alt_v1
            elif os.path.isfile(alt_v2):
                p_voices = alt_v2
        self.kokoro_voices_path = p_voices

        self.cache_dir = resolve_path(self.cache_dir)


def load_audacia_config(yaml_path: str = "config/audacia_config.yaml") -> Dict[str, Any]:
    """Carga configuración desde YAML o retorna defaults tipados."""
    if not os.path.isfile(yaml_path):
        return {
            "tracking": TrackingConfig(),
            "attention": AttentionConfig(),
            "recognition": RecognitionConfig(),
            "interaction": InteractionConfig(),
            "greetings": GreetingsConfig(),
            "audio": AudioConfig()
        }

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    t_data = data.get("tracking", {})
    tracking_cfg = TrackingConfig(
        max_age=t_data.get("max_age", 15),
        min_hits=t_data.get("min_hits", 3),
        iou_threshold=t_data.get("iou_threshold", 0.25),
        distance_threshold=t_data.get("distance_threshold", 0.20)
    )

    a_data = data.get("attention", {})
    attention_cfg = AttentionConfig(
        dwell_candidate_seconds=a_data.get("dwell_candidate_seconds", 0.50),
        dwell_engaging_seconds=a_data.get("dwell_engaging_seconds", 1.20),
        disengage_grace_seconds=a_data.get("disengage_grace_seconds", 0.80),
        candidate_score_threshold=a_data.get("candidate_score_threshold", 0.30),
        engaging_score_threshold=a_data.get("engaging_score_threshold", 0.60),
        min_orientation_for_engaging=a_data.get("min_orientation_for_engaging", 0.60),
        frontal_orientation_grace_seconds=a_data.get("frontal_orientation_grace_seconds", 0.30),
        min_face_height=a_data.get("min_face_height", 70),
        optimal_face_height_min=a_data.get("optimal_face_height_min", 100),
        optimal_face_height_max=a_data.get("optimal_face_height_max", 260),
        max_face_height=a_data.get("max_face_height", 380),
        central_zone_min_x=a_data.get("central_zone_min_x", 0.20),
        central_zone_max_x=a_data.get("central_zone_max_x", 0.80),
        orientation_tolerance_sigma=a_data.get("orientation_tolerance_sigma", 0.35)
    )

    r_data = data.get("recognition", {})
    recognition_cfg = RecognitionConfig(
        similarity_threshold=r_data.get("similarity_threshold", 0.55),
        ambiguity_margin=r_data.get("ambiguity_margin", 0.08),
        samples_for_consolidation=r_data.get("samples_for_consolidation", 3),
        model_path=r_data.get("model_path", "models/face_recognition_sface_2021dec.onnx"),
        db_path=r_data.get("db_path", "data/identities.db")
    )

    i_data = data.get("interaction", {})
    interaction_cfg = InteractionConfig(
        group_stabilization_seconds=i_data.get("group_stabilization_seconds", 0.80),
        identity_resolution_timeout_seconds=i_data.get("identity_resolution_timeout_seconds", 2.0),
        session_timeout_seconds=i_data.get("session_timeout_seconds", 2.5),
        person_cooldown_seconds=i_data.get("person_cooldown_seconds", 45.0),
        unknown_cooldown_seconds=i_data.get("unknown_cooldown_seconds", 25.0),
        allow_late_joiner_greeting=i_data.get("allow_late_joiner_greeting", False)
    )

    g_data = data.get("greetings", {})
    greetings_cfg = GreetingsConfig(
        single_known=g_data.get("single_known", "Hola, bienvenido a AudacIA, {name}."),
        multiple_known=g_data.get("multiple_known", "Hola, bienvenidos a AudacIA, {names}."),
        single_unknown=g_data.get("single_unknown", "Bienvenido a AudacIA."),
        multiple_unknown=g_data.get("multiple_unknown", "Hola, bienvenidos a AudacIA."),
        mixed_group=g_data.get("mixed_group", "Hola, bienvenidos a AudacIA.")
    )

    au_data = data.get("audio", {})
    default_overrides = {
        "AudacIA": "Audacia",
        "Jonathan": "Yónatan",
        "Joshua": "Yóshua",
    }
    audio_cfg = AudioConfig(
        enabled=au_data.get("enabled", True),
        tts_engine=au_data.get("tts_engine", "kokoro"),
        kokoro_voice=au_data.get("kokoro_voice", "ef_dora"),
        kokoro_language=au_data.get("kokoro_language", "es"),
        kokoro_model_path=au_data.get("kokoro_model_path", "scratch/models/kokoro/kokoro-v1.0.onnx"),
        kokoro_voices_path=au_data.get("kokoro_voices_path", "scratch/models/kokoro/voices-v1.0.bin"),
        kokoro_model_version=au_data.get("kokoro_model_version", "kokoro-v1.0"),
        cache_enabled=au_data.get("cache_enabled", True),
        cache_dir=au_data.get("cache_dir", "data/audio_cache"),
        lazy_load=au_data.get("lazy_load", True),
        fallback_to_native=au_data.get("fallback_to_native", True),
        voice_rate=au_data.get("voice_rate", 155),
        voice_volume=au_data.get("voice_volume", 1.0),
        voice_language_preference=au_data.get("voice_language_preference", "es"),
        pronunciation_overrides=au_data.get("pronunciation_overrides", default_overrides)
    )

    c_data = data.get("camera", {})
    camera_cfg = CameraConfig(
        target_width=c_data.get("target_width", 1920),
        target_height=c_data.get("target_height", 1080),
        target_fps=c_data.get("target_fps", 30),
        use_threaded_reader=c_data.get("use_threaded_reader", True),
        max_inference_dimension=c_data.get("max_inference_dimension", 1280)
    )

    d_data = data.get("detection", {})
    detection_cfg = DetectionConfig(
        model_path=d_data.get("model_path", "models/face_detection_yunet_2023mar.onnx"),
        score_threshold=d_data.get("score_threshold", 0.65),
        nms_threshold=d_data.get("nms_threshold", 0.30),
        min_face_size=d_data.get("min_face_size", 55)
    )

    return {
        "camera": camera_cfg,
        "detection": detection_cfg,
        "tracking": tracking_cfg,
        "attention": attention_cfg,
        "recognition": recognition_cfg,
        "interaction": interaction_cfg,
        "greetings": greetings_cfg,
        "audio": audio_cfg,
        "raw": data
    }
