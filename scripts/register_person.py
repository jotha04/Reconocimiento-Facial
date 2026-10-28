"""
Herramienta interactiva de Registro de Personas para AudacIA (Fase 5).
Permite:
1. Registrar persona desde fotografía estática.
2. Registrar persona usando la cámara Logitech C270 con preview en vivo y validación de calidad.
3. Listar identidades registradas en SQLite.
4. Eliminar identidades.
"""

import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import cv2

from audacia.app.camera.device_discovery import find_c270_camera
from audacia.app.camera.camera_stream import CameraStream
from audacia.app.detection.yunet_detector import YuNetDetector
from audacia.app.recognition.sface_recognizer import SFaceRecognizer
from audacia.app.storage.identity_store import IdentityStore
from audacia.app.recognition.registration_service import RegistrationService
from audacia.app.attention.config import load_audacia_config


def register_via_camera():
    cfg_all = load_audacia_config("config/audacia_config.yaml")
    cam_cfg = cfg_all["camera"]

    print("=" * 65)
    print("   REGISTRO DE PERSONA CON CÁMARA HD (LOGITECH C920 / C270)    ")
    print("=" * 65)

    detector = YuNetDetector(
        model_path="models/face_detection_yunet_2023mar.onnx",
        input_size=(cam_cfg.target_width, cam_cfg.target_height),
        min_face_size=50,
        max_inference_dimension=cam_cfg.max_inference_dimension
    )
    recognizer = SFaceRecognizer(model_path="models/face_recognition_sface_2021dec.onnx")
    store = IdentityStore("data/identities.db")
    service = RegistrationService(detector, recognizer, store)

    cam_info = find_c270_camera()
    camera = CameraStream(
        device_info=cam_info,
        target_width=cam_cfg.target_width,
        target_height=cam_cfg.target_height,
        target_fps=cam_cfg.target_fps,
        use_threaded_reader=cam_cfg.use_threaded_reader
    )
    camera.open()
    time.sleep(0.5)

    window_name = f"AudacIA - Registro con Cámara {cam_info.name} (ESPACIO para capturar)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 600)

    print("\n[INSTRUCCIONES]")
    print("  1. Ubícate frente a la cámara mirando de frente.")
    print("  2. Cuando el recuadro esté VERDE, presiona la tecla ESPACIO para capturar.")
    print("  3. Presiona ESC o 'q' para cancelar.\n")

    captured_frame = None

    try:
        while True:
            ret, frame, _ = camera.read(timeout=0.5)
            if not ret or frame is None:
                continue

            vis = frame.copy()
            faces, _ = detector.detect(frame)

            # Guía visual
            status_text = "Buscando rostro..."
            status_color = (0, 165, 255) # Naranja

            if len(faces) == 1:
                f = faces[0]
                q = recognizer.validate_registration_quality(
                    frame, f.bbox, f.confidence, f.orientation_frontal_score,
                    min_height=90, min_confidence=0.65, min_frontal_score=0.60,
                    min_sharpness=30.0
                )
                if q.is_valid:
                    status_text = "Calidad optima! Presiona ESPACIO para capturar"
                    status_color = (46, 204, 113) # Verde
                else:
                    status_text = f"Ajustar: {q.reason}"
                    status_color = (40, 205, 245) # Amarillo

                # Dibujar caja y landmarks
                cv2.rectangle(vis, (f.x, f.y), (f.x + f.w, f.y + f.h), status_color, 2)
                for pt in f.landmarks.astype(int):
                    cv2.circle(vis, tuple(pt), 3, (0, 255, 255), -1)

            elif len(faces) > 1:
                status_text = f"Mas de un rostro detectado ({len(faces)}). Debe haber solo 1 persona."
                status_color = (50, 50, 230) # Rojo

            # Barra de estado inferior dinámica
            h_vis, w_vis = vis.shape[:2]
            cv2.rectangle(vis, (0, h_vis - 40), (w_vis, h_vis), (20, 20, 20), -1)
            cv2.putText(vis, status_text, (15, h_vis - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == 32: # Barra espaciadora
                if len(faces) == 1 and q.is_valid:
                    captured_frame = frame.copy()
                    break
                else:
                    print("No se puede capturar aún: " + status_text)
            elif key in (27, ord('q')):
                print("Registro cancelado por el usuario.")
                return

    finally:
        camera.close()
        cv2.destroyAllWindows()

    if captured_frame is not None:
        print("\n" + "-" * 65)
        print("¡Foto capturada con éxito! Calidad validada.")
        name = input("¿Cómo te llamas? (Ingresa el nombre de la persona): ").strip()
        if not name:
            print("Registro cancelado: nombre vacío.")
            return

        res = service.register_from_frame(captured_frame, name, source_type="camera")
        print("\n" + "=" * 65)
        if res.success:
            print(f"✓ {res.message}")
            print(f"ID asignado: {res.person_id}")
            # Pre-generar y cachear saludo oficial de la nueva persona
            try:
                from audacia.app.audio.tts_engine import TTSEngine
                tts_cfg = cfg_all.get("audio")
                if tts_cfg and tts_cfg.enabled and tts_cfg.cache_enabled:
                    print(f"Pre-generando caché de saludo para '{name}'...")
                    tts_eng = TTSEngine(tts_cfg)
                    tts_eng.initialize()
                    g_text = f"Hola, bienvenido a AudacIA, {name}."
                    art = tts_eng.get_audio_artifact(g_text)
                    if art:
                        print(f"✓ Saludo oficial pre-cacheado: '{g_text}'")
            except Exception as e:
                print(f"[AVISO AUDIO] No se pudo pre-cachear saludo: {e}")
        else:
            print(f"✗ Falló el registro: {res.message}")
        print("=" * 65 + "\n")


def register_via_photo(image_path: str, name: str):
    print("=" * 65)
    print(f"    REGISTRO DESDE FOTOGRAFÍA: '{image_path}'    ")
    print("=" * 65)

    detector = YuNetDetector(model_path="models/face_detection_yunet_2023mar.onnx", min_face_size=40)
    recognizer = SFaceRecognizer(model_path="models/face_recognition_sface_2021dec.onnx")
    store = IdentityStore("data/identities.db")
    service = RegistrationService(detector, recognizer, store)

    res = service.register_from_image_path(image_path, name)
    if res.success:
        print(f"✓ {res.message}")
        print(f"ID: {res.person_id}")
        try:
            from audacia.app.attention.config import load_audacia_config
            from audacia.app.audio.tts_engine import TTSEngine
            cfg_all = load_audacia_config("config/audacia_config.yaml")
            tts_cfg = cfg_all.get("audio")
            if tts_cfg and tts_cfg.enabled and tts_cfg.cache_enabled:
                tts_eng = TTSEngine(tts_cfg)
                tts_eng.initialize()
                g_text = f"Hola, bienvenido a AudacIA, {name}."
                art = tts_eng.get_audio_artifact(g_text)
                if art:
                    print(f"✓ Saludo oficial pre-cacheado: '{g_text}'")
        except Exception as e:
            print(f"[AVISO AUDIO] No se pudo pre-cachear saludo: {e}")
    else:
        print(f"✗ Falló el registro: {res.message}")


def list_identities():
    store = IdentityStore("data/identities.db")
    persons = store.list_persons()
    print("\n" + "=" * 65)
    print(f"       CATÁLOGO DE IDENTIDADES EN SQLITE ({len(persons)} personas)       ")
    print("=" * 65)
    if not persons:
        print("  (No hay personas registradas en la base de datos)")
    for p in persons:
        print(f"  - Nombre: {p['name']:<20} | Muestras: {p['sample_count']} | ID: {p['person_id']}")
    print("=" * 65 + "\n")


def delete_identity(person_id: str):
    store = IdentityStore("data/identities.db")
    if store.delete_person(person_id):
        print(f"✓ Persona con ID '{person_id}' eliminada exitosamente.")
    else:
        print(f"✗ No se encontró ninguna persona con ID '{person_id}'.")


def main():
    parser = argparse.ArgumentParser(description="Gestor de Registro de Identidades - AudacIA")
    parser.add_argument("--camera", action="store_true", help="Iniciar registro interactivo con cámara C270")
    parser.add_argument("--photo", type=str, help="Ruta de la fotografía para registro")
    parser.add_argument("--name", type=str, help="Nombre de la persona a registrar")
    parser.add_argument("--list", action="store_true", help="Listar identidades registradas")
    parser.add_argument("--delete", type=str, help="ID de la persona a eliminar")

    args = parser.parse_args()

    if args.camera:
        register_via_camera()
    elif args.photo:
        if not args.name:
            print("Error: Debes proporcionar --name junto con --photo.")
        else:
            register_via_photo(args.photo, args.name)
    elif args.list:
        list_identities()
    elif args.delete:
        delete_identity(args.delete)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
