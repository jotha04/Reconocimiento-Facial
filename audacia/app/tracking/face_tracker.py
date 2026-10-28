"""
Módulo principal de FaceTracker: seguimiento multirrostro en tiempo real
con Filtro de Kalman cinemático y asociación espacial IoU + distancia centroidal.
Diseñado para cámara estática a 640x480 @ 30 FPS en CPU pura (< 1 ms de latencia).
"""

import time
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

from audacia.app.detection.yunet_detector import FaceDetection
from audacia.app.tracking.track import Track, TrackState
from audacia.app.tracking.kalman_box_tracker import KalmanBoxTracker
from audacia.app.attention.config import TrackingConfig

logger = logging.getLogger(__name__)


def compute_iou(bb1: Tuple[int, int, int, int], bb2: Tuple[int, int, int, int]) -> float:
    """Calcula la intersección sobre unión (IoU) entre dos bounding boxes (x, y, w, h)."""
    x1_a, y1_a, w_a, h_a = bb1
    x2_a, y2_a = x1_a + w_a, y1_a + h_a

    x1_b, y1_b, w_b, h_b = bb2
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = w_a * h_a
    area_b = w_b * h_b
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_normalized_centroid_distance(
    c1: Tuple[int, int], c2: Tuple[int, int], max_diag: float = 800.0
) -> float:
    """Calcula la distancia euclidiana entre dos centros normalizada por la diagonal del frame."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist = np.sqrt(dx * dx + dy * dy)
    return float(dist / max_diag)


class FaceTracker:
    """
    Rastreador facial temporal con asociación voraz compuerta y filtros cinemáticos de Kalman.
    Tolera pérdidas temporales de detección (dropouts) manteniendo la identidad temporal Track #ID.
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        frame_width: float = 1920.0,
        frame_height: float = 1080.0
    ):
        self.config = config or TrackingConfig()
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.kalman_trackers: Dict[int, KalmanBoxTracker] = {}
        self.set_dimensions(frame_width, frame_height)

    def set_dimensions(self, width: float, height: float):
        import math
        self.frame_width = max(1.0, float(width))
        self.frame_height = max(1.0, float(height))
        self.frame_diagonal = math.hypot(self.frame_width, self.frame_height)

    def update(self, detections: List[FaceDetection]) -> List[Track]:
        """
        Paso de actualización del tracker:
        1. Predice la posición de todos los tracks activos con su filtro de Kalman.
        2. Asocia detecciones con tracks existentes mediante IoU y distancia entre centros.
        3. Actualiza tracks coincidentes con nuevas mediciones.
        4. Inicializa nuevos tracks para detecciones no asociadas.
        5. Purga tracks que han superado el límite de pérdidas consecutivas (max_age).
        """
        now = time.monotonic()
        track_ids = list(self.tracks.keys())

        # Auto-adaptar dimensiones si las detecciones exceden el encuadre actual
        for det in detections:
            max_x = det.bbox[0] + det.bbox[2]
            max_y = det.bbox[1] + det.bbox[3]
            if max_x > self.frame_width or max_y > self.frame_height:
                self.set_dimensions(max(self.frame_width, max_x * 1.1), max(self.frame_height, max_y * 1.1))

        # 1. Predicción cinemática para cada track activo
        predicted_boxes = {}
        for tid in track_ids:
            pred_box = self.kalman_trackers[tid].predict()
            predicted_boxes[tid] = pred_box
            track = self.tracks[tid]
            track.age_frames += 1
            track.consecutive_misses += 1

        num_tracks = len(track_ids)
        num_dets = len(detections)

        matched_tracks = set()
        matched_dets = set()

        # 2. Asociación Bipartita en Dos Fases:
        # FASE 1: Emparejamiento Geométrico Primario de Alto IoU (Prioridad Absoluta)
        # Asegura que cada persona retenga su propio recuadro facial entre fotogramas consecutivos.
        if num_tracks > 0 and num_dets > 0:
            stage1_candidates = []
            for t_idx, tid in enumerate(track_ids):
                pred_bb = predicted_boxes[tid]
                pred_center = (pred_bb[0] + pred_bb[2] // 2, pred_bb[1] + pred_bb[3] // 2)
                track = self.tracks[tid]
                is_known = (track.identity_decision == "KNOWN")

                for d_idx, det in enumerate(detections):
                    det_bb = det.bbox
                    det_center = det.center

                    iou = compute_iou(pred_bb, det_bb)
                    dist = compute_normalized_centroid_distance(pred_center, det_center, self.frame_diagonal)

                    # Consistencia de escala facial: ancho y alto no pueden fluctuar bruscamente en 1 frame
                    scale_w = det.w / max(1, track.width)
                    scale_h = det.h / max(1, track.height)
                    scale_ok = (0.55 <= scale_w <= 1.80) and (0.55 <= scale_h <= 1.80)

                    # En Fase 1: Requiere solapamiento geométrico real
                    # Para tracks KNOWN se exige un solapamiento mínimo de 0.20 para que un transeúnte no le robe el ID
                    min_iou = 0.20 if is_known else 0.15
                    if scale_ok and iou >= min_iou:
                        # Costo dominado fuertemente por IoU; menor costo = mejor match
                        cost = (1.0 - iou) * 2.0 + dist
                        stage1_candidates.append((cost, tid, d_idx))

            stage1_candidates.sort(key=lambda x: x[0])
            for cost, tid, d_idx in stage1_candidates:
                if tid not in matched_tracks and d_idx not in matched_dets:
                    matched_tracks.add(tid)
                    matched_dets.add(d_idx)
                    det = detections[d_idx]
                    track = self.tracks[tid]
                    kf = self.kalman_trackers[tid]

                    vx, vy = kf.update(det.bbox)
                    track.bbox = det.bbox
                    track.center = det.center
                    track.width = det.w
                    track.height = det.h
                    track.landmarks = det.landmarks
                    track.head_orientation_proxy = det.head_orientation_proxy
                    track.orientation_frontal_score = det.orientation_frontal_score
                    track.raw_detection = det.raw_detection
                    track.is_complete_face = getattr(det, 'is_complete_face', True)
                    track.velocity_x = round(vx, 2)
                    track.velocity_y = round(vy, 2)
                    track.last_seen = now
                    track.frames_seen += 1
                    track.consecutive_hits += 1
                    track.consecutive_misses = 0

            # FASE 2: Emparejamiento Secundario de Proximidad para Tracks No Asociados
            # Aplica compuertas estrictas para no robar identidades de personas cercanas
            unmatched_tids = [tid for tid in track_ids if tid not in matched_tracks]
            unmatched_dids = [d_idx for d_idx in range(num_dets) if d_idx not in matched_dets]

            if unmatched_tids and unmatched_dids:
                stage2_candidates = []
                for tid in unmatched_tids:
                    track = self.tracks[tid]
                    is_known = (track.identity_decision == "KNOWN")
                    pred_bb = predicted_boxes[tid]
                    pred_center = (pred_bb[0] + pred_bb[2] // 2, pred_bb[1] + pred_bb[3] // 2)

                    # Si el track ya es KNOWN, NO asociar en proximidad laxa sin IoU
                    # Es preferible coasting de Kalman por breves pérdidas que saltar a un transeúnte
                    max_dist = 0.06 if is_known else 0.10
                    min_iou = 0.10 if is_known else 0.0

                    for d_idx in unmatched_dids:
                        det = detections[d_idx]
                        iou = compute_iou(pred_bb, det.bbox)
                        dist = compute_normalized_centroid_distance(pred_center, det.center, self.frame_diagonal)

                        scale_w = det.w / max(1, track.width)
                        scale_h = det.h / max(1, track.height)
                        scale_ok = (0.65 <= scale_w <= 1.50) and (0.65 <= scale_h <= 1.50)

                        if scale_ok and dist <= max_dist and iou >= min_iou:
                            cost = dist * 2.0 + (1.0 - iou)
                            stage2_candidates.append((cost, tid, d_idx))

                stage2_candidates.sort(key=lambda x: x[0])
                for cost, tid, d_idx in stage2_candidates:
                    if tid not in matched_tracks and d_idx not in matched_dets:
                        matched_tracks.add(tid)
                        matched_dets.add(d_idx)
                        det = detections[d_idx]
                        track = self.tracks[tid]
                        kf = self.kalman_trackers[tid]

                        vx, vy = kf.update(det.bbox)
                        track.bbox = det.bbox
                        track.center = det.center
                        track.width = det.w
                        track.height = det.h
                        track.landmarks = det.landmarks
                        track.head_orientation_proxy = det.head_orientation_proxy
                        track.orientation_frontal_score = det.orientation_frontal_score
                        track.raw_detection = det.raw_detection
                        track.is_complete_face = getattr(det, 'is_complete_face', True)
                        track.velocity_x = round(vx, 2)
                        track.velocity_y = round(vy, 2)
                        track.last_seen = now
                        track.frames_seen += 1
                        track.consecutive_hits += 1
                        track.consecutive_misses = 0

        # 4. Tratar tracks no emparejados (missed)
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.consecutive_hits = 0
                # Actualizar bbox con la predicción de Kalman para mantener continuidad
                pred_bb = predicted_boxes[tid]
                track.bbox = pred_bb
                track.center = (pred_bb[0] + pred_bb[2] // 2, pred_bb[1] + pred_bb[3] // 2)
                track.width = pred_bb[2]
                track.height = pred_bb[3]

        # 5. Crear nuevos tracks para detecciones no asociadas (con prevención estricta de duplicados)
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                # Anti-fantasma estricto: Solo rescatar si el track no emparejado tiene solapamiento geométrico real
                # (IoU >= 0.15 y distancia estrecha <= 0.08) y NUNCA a un track KNOWN a menos que IoU >= 0.25.
                best_unmatched_tid = None
                best_dist = 0.08
                for tid in track_ids:
                    if tid not in matched_tracks and tid in self.tracks:
                        trk = self.tracks[tid]
                        iou_rescue = compute_iou(trk.bbox, det.bbox)
                        min_rescue_iou = 0.25 if trk.identity_decision == "KNOWN" else 0.12
                        if iou_rescue < min_rescue_iou:
                            continue
                        t_center = trk.center
                        d = compute_normalized_centroid_distance(t_center, det.center, self.frame_diagonal)
                        if d < best_dist:
                            best_dist = d
                            best_unmatched_tid = tid

                if best_unmatched_tid is not None:
                    matched_tracks.add(best_unmatched_tid)
                    track = self.tracks[best_unmatched_tid]
                    kf = self.kalman_trackers[best_unmatched_tid]
                    vx, vy = kf.update(det.bbox)
                    track.bbox = det.bbox
                    track.center = det.center
                    track.width = det.w
                    track.height = det.h
                    track.landmarks = det.landmarks
                    track.head_orientation_proxy = det.head_orientation_proxy
                    track.orientation_frontal_score = det.orientation_frontal_score
                    track.raw_detection = det.raw_detection
                    track.is_complete_face = getattr(det, 'is_complete_face', True)
                    track.velocity_x = round(vx, 2)
                    track.velocity_y = round(vy, 2)
                    track.last_seen = now
                    track.frames_seen += 1
                    track.consecutive_hits += 1
                    track.consecutive_misses = 0
                    continue

                new_id = self.next_track_id
                self.next_track_id += 1

                new_track = Track(
                    track_id=new_id,
                    created_at=now,
                    last_seen=now,
                    bbox=det.bbox,
                    center=det.center,
                    width=det.w,
                    height=det.h,
                    landmarks=det.landmarks,
                    head_orientation_proxy=det.head_orientation_proxy,
                    orientation_frontal_score=det.orientation_frontal_score,
                    raw_detection=det.raw_detection,
                    is_complete_face=getattr(det, 'is_complete_face', True),
                    state=TrackState.OBSERVED
                )
                self.tracks[new_id] = new_track
                self.kalman_trackers[new_id] = KalmanBoxTracker(det.bbox)

        # 6. Eliminar tracks caducados (pérdidas que superan max_age)
        expired_ids = [
            tid for tid, trk in self.tracks.items()
            if trk.consecutive_misses > self.config.max_age
        ]
        for tid in expired_ids:
            # Marcar como DISENGAGED antes de purgar
            self.tracks[tid].state = TrackState.DISENGAGED
            del self.tracks[tid]
            del self.kalman_trackers[tid]

        # Solo retornar pistas confirmadas (previene artefactos efímeros al mover la cámara)
        return [trk for trk in self.tracks.values() if trk.is_confirmed]
