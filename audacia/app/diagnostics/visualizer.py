"""
Módulo de visualización y renderizado de diagnósticos en pantalla (OpenCV GUI).
Actualizado para Fase 5: Muestra explícitamente:
- Detection confidence
- Recognition similarity
- Identity decision (NOT_EVALUATED, KNOWN, UNKNOWN)
- Badges limpios sobre cada rostro y panel lateral de foco no invasivo.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from audacia.app.tracking.track import Track, TrackState


class DiagnosticVisualizer:
    """
    Renderiza anotaciones claras, identificaciones faciales y telemetría
    sin obstaculizar la visión del entorno.
    """

    def __init__(self):
        # Paleta semántica BGR
        self.COLOR_OBSERVED = (235, 170, 50)       # Azul / Cian claro
        self.COLOR_CANDIDATE = (40, 205, 245)     # Ámbar / Amarillo brillante
        self.COLOR_ENGAGING = (46, 204, 113)      # Verde esmeralda brillante
        self.COLOR_DISENGAGED = (120, 120, 130)   # Gris tenue
        self.COLOR_KNOWN = (50, 255, 120)         # Verde neón para persona reconocida
        self.COLOR_UNKNOWN = (80, 120, 255)       # Naranja/Rojo suave para desconocido
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (18, 18, 22)
        self.COLOR_PANEL_BG = (25, 26, 32)
        self.COLOR_CYAN = (230, 180, 50)
        self.COLOR_MAGENTA = (180, 70, 200)

    def get_state_color(self, state: TrackState, identity_decision: str = "NOT_EVALUATED") -> tuple:
        if state == TrackState.ENGAGING:
            if identity_decision == "KNOWN":
                return self.COLOR_KNOWN
            elif identity_decision == "UNKNOWN":
                return self.COLOR_UNKNOWN
            elif identity_decision in ("IDENTIFYING", "NOT_EVALUATED"):
                return self.COLOR_CYAN
            return self.COLOR_ENGAGING
        elif state == TrackState.ATTENTION_CANDIDATE:
            if identity_decision == "KNOWN":
                return self.COLOR_KNOWN
            elif identity_decision == "IDENTIFYING":
                return self.COLOR_CYAN
            return self.COLOR_CANDIDATE
        elif state == TrackState.OBSERVED:
            return self.COLOR_OBSERVED
        else:
            return self.COLOR_DISENGAGED

    def render(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        metrics_summary: Dict[str, Any],
        camera_telemetry: Dict[str, Any],
        known_count: int = 0,
        interaction_telemetry: Optional[Dict[str, Any]] = None,
        show_breakdown: bool = True
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        primary_track: Optional[Track] = None
        max_score = -1.0

        # 1. Dibujar cada rostro de manera compacta y limpia
        for track in tracks:
            # Desaparecer inmediatamente si el rostro fue ocluido o cubierto (evitar derivas o recuadros fantasma)
            if track.consecutive_misses > 0:
                continue

            state_color = self.get_state_color(track.state, track.identity_decision)
            x, y, bw, bh = track.bbox

            if track.attention_score > max_score:
                max_score = track.attention_score
                primary_track = track

            # Coordenadas acotadas
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            bw = max(1, min(w - x, bw))
            bh = max(1, min(h - y, bh))

            # Bounding Box delgado (2px)
            thickness = 2 if track.state == TrackState.ENGAGING else 1
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), state_color, thickness)

            # Landmarks faciales discretos
            if track.landmarks is not None and len(track.landmarks) == 5:
                lm = track.landmarks.astype(int)
                r_eye, l_eye, nose = lm[0], lm[1], lm[2]
                r_m, l_m = lm[3], lm[4]

                cv2.circle(vis, tuple(r_eye), 2, self.COLOR_CYAN, -1)
                cv2.circle(vis, tuple(l_eye), 2, self.COLOR_CYAN, -1)
                cv2.circle(vis, tuple(nose), 2, (0, 255, 255), -1)
                cv2.circle(vis, tuple(r_m), 2, self.COLOR_MAGENTA, -1)
                cv2.circle(vis, tuple(l_m), 2, self.COLOR_MAGENTA, -1)

                eye_mid = ((r_eye[0] + l_eye[0]) // 2, (r_eye[1] + l_eye[1]) // 2)
                cv2.line(vis, eye_mid, tuple(nose), state_color, 1)

            # Badge superior ultra-compacto (solo 16px de altura justo sobre la caja)
            if track.identity_decision == "KNOWN":
                badge_text = f"#{track.track_id} {track.identity_label} [{track.recognition_similarity:.2f}]"
            elif track.identity_decision == "UNKNOWN":
                badge_text = f"#{track.track_id} UNKNOWN [{track.recognition_similarity:.2f}]"
            elif track.identity_decision == "IDENTIFYING":
                badge_text = f"#{track.track_id} {track.identity_label} (Verificando...)"
            elif getattr(track, 'recognition_candidate_history', None) and len(track.recognition_candidate_history) > 0:
                cand = track.recognition_candidate_history[-1]
                if cand[0] == "KNOWN":
                    badge_text = f"#{track.track_id} {cand[1]} [{cand[2]:.2f}]"
                else:
                    badge_text = f"#{track.track_id} Identificando..."
            else:
                badge_text = f"#{track.track_id} [{track.state.value[:3]} {track.attention_score:.2f}]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(badge_text, font, 0.40, 1)

            badge_y = max(th + 4, y - 4)
            badge_bg_y1 = max(0, badge_y - th - 3)
            badge_bg_y2 = badge_y + 3

            cv2.rectangle(vis, (x, badge_bg_y1), (x + tw + 6, badge_bg_y2), state_color, -1)
            cv2.putText(vis, badge_text, (x + 3, badge_y), font, 0.40, self.COLOR_BLACK, 1, cv2.LINE_AA)

        # 2. Panel lateral limpio en esquina superior derecha (solo para el track en foco)
        if show_breakdown and primary_track is not None and primary_track.height >= 50:
            self._draw_focus_card(vis, primary_track, w)

        # 3. Banner Superior Compacto (HUD de Telemetría)
        self._draw_header_hud(vis, tracks, metrics_summary, camera_telemetry, known_count, w)

        # 4. Tarjeta Inferior de Sesión de Interacción y Saludo (Fase 6)
        self._draw_interaction_card(vis, interaction_telemetry, w, h)

        return vis

    def _draw_focus_card(self, vis: np.ndarray, track: Track, frame_w: int):
        """Dibuja una única tarjeta compacta en la esquina superior derecha con los 3 conceptos separados."""
        state_color = self.get_state_color(track.state, track.identity_decision)
        card_w = 230
        card_h = 108
        card_x = frame_w - card_w - 10
        card_y = 65

        overlay = vis.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), self.COLOR_PANEL_BG, -1)
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), state_color, 1)
        cv2.addWeighted(overlay, 0.88, vis, 0.12, 0, vis)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Cabecera de Track y Estado
        line1 = f"FOCUS: Track #{track.track_id} [{track.state.value}]"
        # 2. Identidad y Decisión
        line2 = f"Identity: {track.identity_label} ({track.identity_decision})"
        # 3. Recognition similarity separada de detection
        line3 = f"Recog Sim: {track.recognition_similarity:.2f} | Samples: {track.recognition_eval_count}/3"
        # 4. Attention score y dwell time
        line4 = f"Attention: {track.attention_score:.2f} | Dwell: {track.dwell_time:.1f}s"
        # 5. Geometría
        line5 = f"Size: {track.width}x{track.height}px | Orient: {track.orientation_frontal_score:.2f}"

        cv2.putText(vis, line1, (card_x + 8, card_y + 16), font, 0.38, state_color, 1, cv2.LINE_AA)
        cv2.putText(vis, line2, (card_x + 8, card_y + 34), font, 0.38, self.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(vis, line3, (card_x + 8, card_y + 52), font, 0.36, (140, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, line4, (card_x + 8, card_y + 70), font, 0.36, self.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(vis, line5, (card_x + 8, card_y + 88), font, 0.34, (180, 180, 180), 1, cv2.LINE_AA)

        if track.attention_components:
            comp = track.attention_components
            line6 = f"P:{comp.get('persistence', 0):.1f} S:{comp.get('size', 0):.1f} Pos:{comp.get('position', 0):.1f} O:{comp.get('orientation', 0):.1f}"
            cv2.putText(vis, line6, (card_x + 8, card_y + 102), font, 0.28, (120, 200, 240), 1, cv2.LINE_AA)

    def _draw_header_hud(
        self,
        vis: np.ndarray,
        tracks: List[Track],
        metrics_summary: Dict[str, Any],
        camera_telemetry: Dict[str, Any],
        known_count: int,
        frame_w: int
    ):
        """Banner superior sin colisiones de texto."""
        banner_h = 58
        frame_h = vis.shape[0]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, banner_h), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.88, vis, 0.12, 0, vis)

        fps_eff = metrics_summary.get("effective_fps", 0.0)
        fourcc = camera_telemetry.get("fourcc", "YUY2")

        eng_count = sum(1 for t in tracks if t.state == TrackState.ENGAGING)
        known_in_scene = sum(1 for t in tracks if t.identity_decision == "KNOWN")

        # Línea 1: Cámara + Catálogo de identidades
        cam_raw_name = camera_telemetry.get("name", "Camera")
        if "c920" in cam_raw_name.lower():
            cam_label = "LOGI C920 PRO HD"
        elif "c270" in cam_raw_name.lower():
            cam_label = "LOGI C270 HD"
        elif "face" in cam_raw_name.lower():
            cam_label = "FACETIME HD"
        else:
            cam_label = cam_raw_name[:18].upper()

        t_cam = f"{cam_label} | {frame_w}x{frame_h} @ {fps_eff:.1f} FPS [{fourcc}]"
        t_db = f"DB IDENTITIES: {known_count} | RECOGNIZED IN SCENE: {known_in_scene}"
        cv2.putText(vis, t_cam, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_WHITE, 1, cv2.LINE_AA)
        db_x = max(340, int(frame_w * 0.45))
        cv2.putText(vis, t_db, (db_x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 255, 160), 1, cv2.LINE_AA)

        # Línea 2: Latencias desglosadas
        det_ms = metrics_summary.get("detector_ms", {}).get("avg", 0.0)
        trk_ms = metrics_summary.get("tracking_ms", {}).get("avg", 0.0)
        att_ms = metrics_summary.get("attention_ms", {}).get("avg", 0.0)
        rec_ms = metrics_summary.get("recognition_ms", {}).get("avg", 0.0)
        pipe_ms = metrics_summary.get("pipeline_ms", {}).get("avg", 0.0)

        t_lat = f"LATENCY: Det={det_ms:.1f}ms | Trk={trk_ms:.2f}ms | Attn={att_ms:.2f}ms | Recog={rec_ms:.2f}ms | Total={pipe_ms:.1f}ms"
        cv2.putText(vis, t_lat, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_CYAN, 1, cv2.LINE_AA)

        # Línea 3: Guía semántica
        t_legend = "VERDE NEÓN: Known Identity | NARANJA: Unknown Face | AZUL: Observed | AMARILLO: Candidate"
        cv2.putText(vis, t_legend, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 170, 180), 1, cv2.LINE_AA)

    def _draw_interaction_card(
        self,
        vis: np.ndarray,
        telemetry: Optional[Dict[str, Any]],
        frame_w: int,
        frame_h: int
    ):
        """Dibuja el panel inferior de la sesión de interacción y el estado de audio/saludo."""
        if not telemetry:
            return

        card_w = min(500, frame_w - 20)
        card_h = 88
        card_x = 10
        card_y = frame_h - card_h - 8

        overlay = vis.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), self.COLOR_PANEL_BG, -1)

        active = telemetry.get("active_session", False)
        state_str = telemetry.get("state", "NO_INTERACTION")

        if state_str == "GREETING_PLAYING":
            border_color = self.COLOR_KNOWN
        elif active:
            border_color = self.COLOR_CYAN
        else:
            border_color = self.COLOR_DISENGAGED

        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), border_color, 1)
        cv2.addWeighted(overlay, 0.88, vis, 0.12, 0, vis)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if not active:
            t1 = "INTERACTION: IDLE (Esperando participantes en ENGAGING...)"
            t2 = f"Total Saludos: {telemetry.get('total_greetings', 0)} | Audio Queue: IDLE"
            cv2.putText(vis, t1, (card_x + 8, card_y + 28), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(vis, t2, (card_x + 8, card_y + 54), font, 0.34, (140, 140, 140), 1, cv2.LINE_AA)
            return

        sid = telemetry.get("session_id", "?")
        gtype = telemetry.get("group_type", "NONE")
        t_header = f"SESSION #{sid}: {state_str} [{gtype}]"
        cv2.putText(vis, t_header, (card_x + 8, card_y + 16), font, 0.38, border_color, 1, cv2.LINE_AA)

        # Línea 2: Greeting Batch y Razón
        batch = telemetry.get("greeting_batch", [])
        batch_str = ", ".join(batch) if batch else "None"
        reason = telemetry.get("batch_reason", "NONE")
        t_batch = f"Batch: [{batch_str}] | Reason: {reason}"
        cv2.putText(vis, t_batch, (card_x + 8, card_y + 33), font, 0.34, (255, 230, 130), 1, cv2.LINE_AA)

        # Línea 3: Texto de saludo
        greet = telemetry.get("greeting_text", "")
        if greet:
            t_greet = f'"{greet}"'
            if len(t_greet) > 65:
                t_greet = t_greet[:62] + '..."'
            cv2.putText(vis, t_greet, (card_x + 8, card_y + 50), font, 0.34, (120, 240, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, '"(Sin saludo activo)"', (card_x + 8, card_y + 50), font, 0.32, (150, 150, 150), 1, cv2.LINE_AA)

        # Línea 4: Cooldowns individuales
        cds = telemetry.get("cooldowns", {})
        if cds:
            cd_str = " | ".join([f"{k}={v}" for k, v in list(cds.items())[:3]])
        else:
            cd_str = "None"
        t_cd = f"Cooldown: {cd_str}"
        cv2.putText(vis, t_cd, (card_x + 8, card_y + 68), font, 0.30, (180, 200, 220), 1, cv2.LINE_AA)

        # Línea 5: Audio y Saludos totales
        is_playing = telemetry.get("is_audio_playing", False)
        audio_status = "REPRODUCIENDO" if is_playing else "READY"
        t_audio = f"Audio: {audio_status} | Saludos: {telemetry.get('total_greetings', 0)}"
        cv2.putText(vis, t_audio, (card_x + 8, card_y + 82), font, 0.28, (160, 210, 180), 1, cv2.LINE_AA)

