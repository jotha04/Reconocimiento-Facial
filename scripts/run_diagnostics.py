"""
Script principal interactivo para la Fase 4 de AudacIA:
Pipeline: C270 (Estática) -> YuNet -> FaceTracker (Kalman + IoU) -> AttentionEngine -> HUD Diagnóstico.
"""

import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import logging
import cv2

# Configurar logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("AudacIA-Fase4")

from audacia.app.camera.device_discovery import find_c270_camera
from audacia.app.camera.camera_stream import CameraStream, MockCameraStream
from audacia.app.detection.yunet_detector import YuNetDetector
from audacia.app.tracking.face_tracker import FaceTracker
from audacia.app.attention.attention_engine import AttentionEngine
from audacia.app.recognition.sface_recognizer import SFaceRecognizer
from audacia.app.storage.identity_store import IdentityStore
from audacia.app.recognition.recognition_engine import RecognitionEngine
from audacia.app.interaction.interaction_engine import InteractionEngine
from audacia.app.attention.config import load_audacia_config
from audacia.app.diagnostics.metrics_tracker import MetricsTracker
from audacia.app.diagnostics.visualizer import DiagnosticVisualizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AudacIA Live Diagnostic Visor")
    parser.add_argument("--mock", "--demo", action="store_true", help="Usar cámara sintética de demostración")
    args, _ = parser.parse_known_args()

    print("=" * 70)
    print("     AUDACIA - VISOR EN VIVO DE INTERACTION & AUDIO GREETING (FASE 6)  ")
    print("=" * 70)

    # 1. Cargar configuración desacoplada (YAML)
    config_dict = load_audacia_config("config/audacia_config.yaml")
    camera_cfg = config_dict["camera"]
    detection_cfg = config_dict.get("detection")
    tracking_cfg = config_dict["tracking"]
    attention_cfg = config_dict["attention"]
    recognition_cfg = config_dict["recognition"]
    interaction_cfg = config_dict["interaction"]
    greetings_cfg = config_dict["greetings"]
    audio_cfg = config_dict["audio"]

    # 2. Descubrimiento de la cámara externa o conmutación a MockCameraStream
    use_mock = args.mock
    c270_info = None

    if not use_mock:
        try:
            c270_info = find_c270_camera()
            print(f"[CÁMARA CONFIRMADA] {c270_info.name} (Índice {c270_info.index}, Cámara ESTÁTICA)")
        except RuntimeError as err:
            print(f"\n[AVISO DE CÁMARA FÍSICA] {err}")
            print("[MODO DEMOSTRACIÓN ACTIVADO] Conmutando a cámara sintética MockCameraStream para ejecución continua...")
            use_mock = True

    # 3. Inicializar componentes del pipeline
    if use_mock:
        camera = MockCameraStream(
            target_width=camera_cfg.target_width,
            target_height=camera_cfg.target_height,
            target_fps=camera_cfg.target_fps
        )
    else:
        camera = CameraStream(
            device_info=c270_info,
            target_width=camera_cfg.target_width,
            target_height=camera_cfg.target_height,
            target_fps=camera_cfg.target_fps,
            use_threaded_reader=camera_cfg.use_threaded_reader
        )

    detector = YuNetDetector(
        model_path=detection_cfg.model_path if detection_cfg else "models/face_detection_yunet_2023mar.onnx",
        input_size=(camera_cfg.target_width, camera_cfg.target_height),
        score_threshold=detection_cfg.score_threshold if detection_cfg else 0.65,
        nms_threshold=detection_cfg.nms_threshold if detection_cfg else 0.30,
        min_face_size=detection_cfg.min_face_size if detection_cfg else 55,
        max_inference_dimension=camera_cfg.max_inference_dimension
    )

    tracker = FaceTracker(
        config=tracking_cfg,
        frame_width=camera_cfg.target_width,
        frame_height=camera_cfg.target_height
    )
    attention_engine = AttentionEngine(
        config=attention_cfg,
        frame_width=camera_cfg.target_width,
        frame_height=camera_cfg.target_height
    )
    
    # Componentes de Fase 5: Identidades locales y SFace
    identity_store = IdentityStore(db_path=recognition_cfg.db_path)
    sface_recognizer = SFaceRecognizer(model_path=recognition_cfg.model_path)
    recognition_engine = RecognitionEngine(
        recognizer=sface_recognizer,
        identity_store=identity_store,
        config=recognition_cfg
    )

    # Componentes de Fase 6: Interaction Decision & Audio Greeting
    interaction_engine = InteractionEngine(
        interaction_config=interaction_cfg,
        greetings_config=greetings_cfg,
        audio_config=audio_cfg
    )

    # Pre-calentar caché de audio para todas las identidades registradas (garantiza 0.2ms en stream)
    if audio_cfg.enabled and audio_cfg.cache_enabled:
        registered = identity_store.list_persons()
        names = [p.get("name", "").strip() for p in registered if p.get("name", "").strip()]
        for p_name in names:
            greeting = greetings_cfg.single_known.format(name=p_name)
            interaction_engine.audio_queue.tts.get_audio_artifact(greeting)

        if len(names) >= 2:
            from audacia.app.interaction.greeting_decision import format_names_spanish
            g_pair1 = greetings_cfg.multiple_known.format(names=format_names_spanish(names[:2]))
            interaction_engine.audio_queue.tts.get_audio_artifact(g_pair1)
            g_pair2 = greetings_cfg.multiple_known.format(names=format_names_spanish(names[:2][::-1]))
            interaction_engine.audio_queue.tts.get_audio_artifact(g_pair2)

    metrics = MetricsTracker(window_size=60)
    visualizer = DiagnosticVisualizer()

    # 4. Iniciar cámara y descartar frames de inicio
    camera.open()
    time.sleep(0.5)

    print("\n[CONTROLES DEL VISOR]")
    print("  'q' o ESC : Salir")
    print("  'd'       : Alternar desglose de señales de atención (P, S, Pos, O, Stb)")
    print("  '+' / '-' : Ajustar umbral de confianza de YuNet (+0.05 / -0.05)")
    print("  'r'       : Recargar identidades desde base de datos SQLite")
    print("  's'       : Guardar captura de diagnóstico con telemetría en data/snapshots/")
    print("  'b'       : Imprimir snapshot de rendimiento en consola")
    print("\nIniciando visualizador en vivo...\n")

    os.makedirs("data/snapshots", exist_ok=True)
    cam_disp = c270_info.name if c270_info else "Logitech C920 HD"
    window_name = f"AudacIA - Phase 6: Interaction & Audio Greeting ({cam_disp})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    show_breakdown = True

    try:
        while True:
            t_frame_start = metrics.record_frame_start()

            # A. Captura (Hardware HD @ 30 FPS)
            ret, frame, cap_ms = camera.read(timeout=0.5)
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # B. Detección Facial (YuNet ONNX)
            faces, det_ms = detector.detect(frame)

            # C. Face Tracking (Kalman + IoU / Distancia Bipartita)
            t_trk0 = time.perf_counter()
            tracks = tracker.update(faces)
            trk_ms = (time.perf_counter() - t_trk0) * 1000.0

            # D. Attention Engine (Persistencia, Escala, Posición, Orientación, Estabilidad)
            t_att0 = time.perf_counter()
            evaluated_tracks = attention_engine.update(tracks)
            att_ms = (time.perf_counter() - t_att0) * 1000.0

            # E. Face Recognition (SFace - Solo se evalúa en tracks en estado ENGAGING)
            t_rec0 = time.perf_counter()
            processed_tracks = recognition_engine.process_tracks(frame, evaluated_tracks)
            rec_ms = (time.perf_counter() - t_rec0) * 1000.0

            # F. Interaction Engine (Sesión, Estabilización de Grupo, Decisión de Saludo, Cooldown, Audio Queue)
            t_int0 = time.perf_counter()
            _ = interaction_engine.update(processed_tracks)
            interaction_telemetry = interaction_engine.get_telemetry()
            int_ms = (time.perf_counter() - t_int0) * 1000.0

            # G. Renderizado de Diagnósticos (HUD + Badges + Focus Card + Interaction Card)
            t_ren0 = time.perf_counter()
            cam_telemetry = camera.get_telemetry()
            summary = metrics.get_summary()
            known_count = len(identity_store.list_persons())

            vis_frame = visualizer.render(
                frame=frame,
                tracks=processed_tracks,
                metrics_summary=summary,
                camera_telemetry=cam_telemetry,
                known_count=known_count,
                interaction_telemetry=interaction_telemetry,
                show_breakdown=show_breakdown
            )
            ren_ms = (time.perf_counter() - t_ren0) * 1000.0

            # G. Mostrar en pantalla
            cv2.imshow(window_name, vis_frame)

            t_pipe_end = time.perf_counter()
            pipe_ms = (t_pipe_end - t_frame_start) * 1000.0

            metrics.record_metrics(
                capture_ms=cap_ms,
                detector_ms=det_ms,
                tracking_ms=trk_ms,
                attention_ms=att_ms,
                render_ms=ren_ms,
                pipeline_ms=pipe_ms,
                recognition_ms=rec_ms
            )

            # Control de teclado
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('d'):
                show_breakdown = not show_breakdown
                print(f"[HUD] Desglose de componentes: {'Activado' if show_breakdown else 'Desactivado'}")
            elif key in (ord('+'), ord('=')):
                detector.score_threshold = min(0.95, detector.score_threshold + 0.05)
                print(f"[YUNET THRESHOLD] Incrementado a: {detector.score_threshold:.2f}")
            elif key in (ord('-'), ord('_')):
                detector.score_threshold = max(0.2, detector.score_threshold - 0.05)
                print(f"[YUNET THRESHOLD] Reducido a: {detector.score_threshold:.2f}")
            elif key == ord('r'):
                recognition_engine.reload_identities()
                print(f"[RELOAD] Base de identidades recargada. Total: {len(identity_store.list_persons())} personas registradas.")
            elif key == ord('s'):
                snap_path = f"data/snapshots/phase5_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, vis_frame)
                print(f"[SNAPSHOT] Guardado con éxito en: {snap_path}")
            elif key == ord('b'):
                s = metrics.get_summary()
                print("\n--- BENCHMARK SNAPSHOT FASE 5 ---")
                print(f"Frames: {s['total_frames']} | FPS Efectivo: {s['effective_fps']} (Máx Teórico: {s['max_theoretical_fps']})")
                print(
                    f"Latencias: Cap={s['capture_ms']['avg']:.1f}ms | Det={s['detector_ms']['avg']:.1f}ms | "
                    f"Trk={s['tracking_ms']['avg']:.2f}ms | Attn={s['attention_ms']['avg']:.2f}ms | "
                    f"Recog={s['recognition_ms']['avg']:.2f}ms | Ren={s['render_ms']['avg']:.1f}ms | Total={s['pipeline_ms']['avg']:.1f}ms"
                )
                print(f"Tracks Activos: {len(tracks)}")
                for t in tracks:
                    print(f"  Track #{t.track_id}: State={t.state.value} | Identity={t.identity_label} ({t.identity_decision}) | RecogSim={t.recognition_similarity:.2f}")
                print()

    except KeyboardInterrupt:
        print("\nInterrupción manual...")

    finally:
        camera.close()
        interaction_engine.audio_queue.stop()
        cv2.destroyAllWindows()
        final_summary = metrics.get_summary()
        print("\n" + "=" * 70)
        print("          RESUMEN FINAL DE LA SESIÓN (FASE 6)          ")
        print("=" * 70)
        print(f"Total frames procesados: {final_summary['total_frames']}")
        print(f"FPS efectivo sostenido: {final_summary['effective_fps']} FPS")
        print(f"Latencia media de Captura (Grab/Read):   {final_summary['capture_ms']['avg']} ms")
        print(f"Latencia media de Detección (YuNet):     {final_summary['detector_ms']['avg']} ms")
        print(f"Latencia media de Tracking (Kalman):     {final_summary['tracking_ms']['avg']} ms")
        print(f"Latencia media de Attention Engine:      {final_summary['attention_ms']['avg']} ms")
        print(f"Latencia media de Face Recognition:      {final_summary['recognition_ms']['avg']} ms")
        print(f"Latencia media de Renderizado GUI/HUD:   {final_summary['render_ms']['avg']} ms")
        print(f"Latencia media Total del Pipeline:       {final_summary['pipeline_ms']['avg']} ms")
        print(f"Total de saludos de audio emitidos:      {interaction_engine.total_greetings_emitted}")
        print("=" * 70)


if __name__ == "__main__":
    main()
