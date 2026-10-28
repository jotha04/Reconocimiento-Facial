import time
import numpy as np
import pytest
from unittest.mock import MagicMock

from audacia.app.detection.yunet_detector import FaceDetection
from audacia.app.tracking.track import Track, TrackState
from audacia.app.tracking.face_tracker import FaceTracker
from audacia.app.recognition.recognition_engine import RecognitionEngine, RecognitionConfig, MatchResult
from audacia.app.interaction.cooldown_manager import CooldownManager
from audacia.app.interaction.session import Participant


def make_det(x, y, w=80, h=80):
    return FaceDetection(
        bbox=(x, y, w, h),
        confidence=0.95,
        landmarks=np.array([[x+20, y+25], [x+60, y+25], [x+40, y+45], [x+25, y+65], [x+55, y+65]]),
        area=w * h,
        head_orientation_proxy=0.50,
        orientation_frontal_score=0.90,
        raw_detection=np.zeros((15, 1)),
        is_complete_face=True
    )


def test_tracker_does_not_swap_known_identity_to_passing_person():
    """Verifica que un transeúnte cercano no robe el recuadro ni la identidad de una persona reconocida."""
    tracker = FaceTracker(frame_width=1920.0, frame_height=1080.0)

    # 1. Persona A en (100, 100) durante 5 fotogramas
    det_a = make_det(100, 100)
    for _ in range(5):
        tracks = tracker.update([det_a])
    assert len(tracks) == 1
    track_a_id = tracks[0].track_id
    tracks[0].identity_decision = "KNOWN"
    tracks[0].identity_label = "Jonathan"

    # 2. Fotogramas con Persona A en (100, 100) y Persona B pasando en (200, 100)
    det_b = make_det(200, 100)
    for _ in range(3):
        tracks = tracker.update([det_a, det_b])
    assert len(tracks) == 2

    # Track A debe seguir en (100, 100) y conservar su identidad
    tr_a = next(t for t in tracks if t.track_id == track_a_id)
    assert tr_a.identity_label == "Jonathan"
    assert tr_a.bbox[0] == 100

    # 3. Fotograma donde Persona A se oculta brevemente (missed) y solo se detecta Persona B en (210, 100)
    det_b_moved = make_det(210, 100)
    tracks = tracker.update([det_b_moved])

    # El track de Jonathan (KNOWN) NO debe saltar sobre Persona B
    # Debe permanecer en su posición predicha o conservarse sin robarse
    if any(t.track_id == track_a_id for t in tracks):
        tr_a = next(t for t in tracks if t.track_id == track_a_id)
        # La caja de Track A no debe haber saltado a (210, 100)
        assert abs(tr_a.bbox[0] - 100) < 30, f"Track de Jonathan saltó a x={tr_a.bbox[0]}"


def test_visitor_reidentification_and_consistency():
    """Verifica que un visitante que sale y regresa mantenga su número de visitante."""
    recognizer = MagicMock()
    identity_store = MagicMock()
    identity_store.get_all_embeddings.return_value = []

    engine = RecognitionEngine(recognizer=recognizer, identity_store=identity_store)

    # Crear embedding aleatorio para visitante 1
    rng = np.random.default_rng(42)
    emb_v1 = rng.standard_normal(128).astype(np.float32)
    emb_v1 /= np.linalg.norm(emb_v1)

    t1 = Track(track_id=1, identity_decision="UNKNOWN", embedding=emb_v1)
    v1_id = engine.resolve_visitor_identity(t1)
    assert v1_id == 1
    assert t1.identity_label == "Visitante #1"

    # Ahora el visitante se va y regresa (nuevo track_id=5, pero mismo rostro con pequeña variación)
    emb_v1_return = emb_v1 + rng.normal(0, 0.05, 128).astype(np.float32)
    emb_v1_return /= np.linalg.norm(emb_v1_return)

    t2 = Track(track_id=5, identity_decision="UNKNOWN", embedding=emb_v1_return)
    v2_id = engine.resolve_visitor_identity(t2)
    assert v2_id == 1, "El visitante recurrente debió re-identificarse como Visitante #1"
    assert t2.identity_label == "Visitante #1"

    # Ahora llega un nuevo visitante completamente diferente
    emb_v2 = rng.standard_normal(128).astype(np.float32)
    emb_v2 /= np.linalg.norm(emb_v2)

    t3 = Track(track_id=6, identity_decision="UNKNOWN", embedding=emb_v2)
    v3_id = engine.resolve_visitor_identity(t3)
    assert v3_id == 2, "El nuevo visitante debió ser asignado como Visitante #2"
    assert t3.identity_label == "Visitante #2"


def test_visitor_cooldown_distinguishes_same_vs_new():
    """Verifica que el cooldown bloquee al mismo visitante pero habilite al nuevo."""
    cd_mgr = CooldownManager(unknown_cooldown_seconds=25.0)

    rng = np.random.default_rng(123)
    emb_v1 = rng.standard_normal(128).astype(np.float32)
    emb_v1 /= np.linalg.norm(emb_v1)

    p1 = Participant(track_id=1, identity_label="Visitante #1", identity_decision="UNKNOWN", visitor_id=1, embedding=emb_v1, center_x=0.5, center_y=0.5)

    # Saludar al visitante 1
    cd_mgr.record_greeting([p1])

    # El visitante 1 regresa como track #4 con visitor_id=1
    p1_return = Participant(track_id=4, identity_label="Visitante #1", identity_decision="UNKNOWN", visitor_id=1, embedding=emb_v1, center_x=0.5, center_y=0.5)
    in_cd_same, status, rem, reason = cd_mgr.check_participant(p1_return)
    assert in_cd_same is True
    assert reason in ("VISITOR_COOLDOWN", "APPEARANCE_COOLDOWN")

    # Un nuevo visitante llega (visitor_id=2, embedding diferente)
    emb_v2 = rng.standard_normal(128).astype(np.float32)
    emb_v2 /= np.linalg.norm(emb_v2)
    p2_new = Participant(track_id=5, identity_label="Visitante #2", identity_decision="UNKNOWN", visitor_id=2, embedding=emb_v2, center_x=0.5, center_y=0.5)
    in_cd_new, status_new, _, _ = cd_mgr.check_participant(p2_new)
    assert in_cd_new is False, "El nuevo visitante NO debe estar bloqueado por cooldown"


def test_footprint_does_not_block_different_visitor_in_same_spot():
    """Verifica que si un nuevo visitante se para en la misma coordenada donde estuvo el anterior,
    su embedding diferente impida que quede bloqueado por huella espacial."""
    cd_mgr = CooldownManager(unknown_cooldown_seconds=25.0)

    rng = np.random.default_rng(999)
    emb_v1 = rng.standard_normal(128).astype(np.float32)
    emb_v1 /= np.linalg.norm(emb_v1)

    p1 = Participant(track_id=1, identity_label="Visitante #1", identity_decision="UNKNOWN", visitor_id=1, embedding=emb_v1, center_x=0.50, center_y=0.50, bbox_height=100)
    cd_mgr.record_greeting([p1])

    # 1 segundo después, llega un visitante DISTINTO y se para exactamente en (0.50, 0.50)
    emb_v2 = rng.standard_normal(128).astype(np.float32)
    emb_v2 /= np.linalg.norm(emb_v2)

    p2 = Participant(track_id=2, identity_label="Visitante #2", identity_decision="UNKNOWN", visitor_id=2, embedding=emb_v2, center_x=0.50, center_y=0.50, bbox_height=100)
    in_cd, status, rem, reason = cd_mgr.check_participant(p2)
    assert in_cd is False, "El nuevo visitante con rostro diferente NO debe ser bloqueado por huella espacial"


def test_adaptive_learning_is_asynchronous():
    """Verifica que _maybe_adaptive_update no bloquee síncronamente el hilo principal."""
    mock_store = MagicMock()
    # Simular una escritura lenta en SQLite (200ms)
    def slow_add(*args, **kwargs):
        time.sleep(0.15)
    mock_store.add_embedding_to_person.side_effect = slow_add

    mock_recognizer = MagicMock()
    config = RecognitionConfig(adaptive_learning_enabled=True, adaptive_cooldown_seconds=1800.0)
    engine = RecognitionEngine(recognizer=mock_recognizer, identity_store=mock_store, config=config)

    track = Track(track_id=1, bbox=(100, 100, 80, 80))
    track.identity_decision = "KNOWN"
    track.person_id = "test-person-123"
    track.identity_label = "Jonathan"
    track.recognition_similarity = 0.65
    track.orientation_frontal_score = 0.95

    feat = np.ones(128, dtype=np.float32)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    engine._maybe_adaptive_update(track, feat, crop)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Debe retornar de inmediato (< 30ms), ejecutándose en background thread
    assert elapsed_ms < 30.0, f"_maybe_adaptive_update tardó {elapsed_ms:.1f}ms síncronos en retornar"


def test_wait_for_new_frame_behavior():
    """Verifica que wait_for_new_frame entregue fotogramas solo cuando se actualicen."""
    from web.pipeline_bridge import PipelineBridge
    bridge = PipelineBridge(use_mock=True)

    # Inicialmente _frame_id es 0, no hay frame
    fid, frame = bridge.wait_for_new_frame(last_seen_id=0, timeout=0.01)
    assert frame is None

    # Simular un frame producido
    with bridge._lock:
        bridge._latest_jpeg = b"test_jpeg_data"
        bridge._frame_id = 1
        bridge._new_frame_event.set()

    # Ahora debe retornar frame_id 1
    fid, frame = bridge.wait_for_new_frame(last_seen_id=0, timeout=0.05)
    assert fid == 1
    assert frame == b"test_jpeg_data"

    # Si ya vimos frame_id 1, una llamada inmediata con timeout bajo retorna None (sin duplicado)
    fid2, frame2 = bridge.wait_for_new_frame(last_seen_id=1, timeout=0.01)
    assert frame2 is None
