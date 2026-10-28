"""
Motor de Reconocimiento Facial (RecognitionEngine) de AudacIA.
Integra:
- Extracción de embeddings 128-D con SFace.
- Comparación contra base de datos local SQLite (IdentityStore).
- Doble compuerta de seguridad anti-falsos positivos (Umbral absoluto + Margen de ambigüedad).
- Consolidación temporal de identidad sobre múltiples fotogramas antes de declarar KNOWN.
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2

from audacia.app.tracking.track import Track, TrackState
from audacia.app.recognition.sface_recognizer import SFaceRecognizer
from audacia.app.storage.identity_store import IdentityStore

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    decision: str           # "KNOWN" o "UNKNOWN"
    label: str              # Nombre de la persona o "UNKNOWN"
    person_id: Optional[str]# UUID si es KNOWN
    similarity: float       # Similitud coseno máxima alcanzada
    is_ambiguous: bool      # True si la diferencia con el 2do mejor candidato es muy pequeña


@dataclass
class RecognitionConfig:
    similarity_threshold: float = 0.55  # Umbral de similitud coseno (configurable, NO hardcodeado)
    ambiguity_margin: float = 0.08      # Margen mínimo de ventaja frente al segundo candidato
    samples_for_consolidation: int = 3  # Muestras consecutivas requeridas para consolidar identidad
    re_eval_interval_seconds: float = 10.0 # Re-evaluación opcional si la persona sigue presente
    adaptive_learning_enabled: bool = True # Auto-adaptación pasiva para cambios graduales
    adaptive_cooldown_seconds: float = 1800.0 # Intervalo mínimo entre auto-actualizaciones (30 min)


class RecognitionEngine:
    """
    Gestiona el reconocimiento facial y la consolidación de identidades sobre tracks activos.
    """

    def __init__(
        self,
        recognizer: SFaceRecognizer,
        identity_store: IdentityStore,
        config: Optional[RecognitionConfig] = None
    ):
        self.recognizer = recognizer
        self.identity_store = identity_store
        self.config = config or RecognitionConfig()

        # Cache de embeddings en memoria para matching ultrarrápido (< 0.1 ms)
        self._cached_embeddings: List[Dict[str, Any]] = []
        self._last_adaptive_update: Dict[str, float] = {}

        # Memoria efímera de visitantes no registrados para consistencia de identidad y numeración
        self._recent_visitors: List[Dict[str, Any]] = []
        self._next_visitor_id: int = 1
        self._visitor_memory_window_seconds: float = 120.0  # 2 minutos de retención

        self.reload_identities()

    def reload_identities(self):
        """Recarga el catálogo de embeddings desde SQLite a memoria."""
        if self.identity_store is not None:
            self._cached_embeddings = self.identity_store.get_all_embeddings()
            logger.info(f"Catálogo de identidades recargado: {len(self._cached_embeddings)} embeddings en memoria.")
        else:
            self._cached_embeddings = []

    def match_embedding(self, query_feat: np.ndarray) -> MatchResult:
        """
        Compara un embedding de consulta contra todas las identidades registradas.
        Aplica doble compuerta: umbral de similitud y margen de ambigüedad.
        """
        if not self._cached_embeddings:
            return MatchResult(
                decision="UNKNOWN",
                label="UNKNOWN",
                person_id=None,
                similarity=0.0,
                is_ambiguous=False
            )

        # Calcular similitud coseno contra cada embedding registrado
        scores: List[Tuple[float, str, str]] = [] # (score, person_id, name)
        for item in self._cached_embeddings:
            ref_feat = item["embedding"]
            sim = self.recognizer.compute_similarity(query_feat, ref_feat)
            scores.append((sim, item["person_id"], item["name"]))

        # Ordenar de mayor a menor similitud
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_pid, best_name = scores[0]

        # 1. Compuerta de Umbral Absoluto
        if best_score < self.config.similarity_threshold:
            return MatchResult(
                decision="UNKNOWN",
                label="UNKNOWN",
                person_id=None,
                similarity=round(best_score, 3),
                is_ambiguous=False
            )

        # 2. Compuerta de Ambigüedad (evitar asignar nombre si dos personas distintas tienen score muy similar)
        # Buscar el mejor score de una persona DIFERENTE
        second_diff_person_score = -1.0
        for s, pid, name in scores[1:]:
            if pid != best_pid:
                second_diff_person_score = s
                break

        is_ambiguous = False
        if second_diff_person_score > 0:
            diff = best_score - second_diff_person_score
            if diff < self.config.ambiguity_margin:
                # Muy reñido entre dos personas distintas -> Por seguridad declarar UNKNOWN
                is_ambiguous = True
                logger.warning(
                    f"Match ambiguo: {best_name} ({best_score:.2f}) vs otro ({second_diff_person_score:.2f}), "
                    f"margen {diff:.2f} < {self.config.ambiguity_margin}. Declarando UNKNOWN por seguridad."
                )
                return MatchResult(
                    decision="UNKNOWN",
                    label="UNKNOWN",
                    person_id=None,
                    similarity=round(best_score, 3),
                    is_ambiguous=True
                )

        # Match exitoso y seguro
        return MatchResult(
            decision="KNOWN",
            label=best_name,
            person_id=best_pid,
            similarity=round(best_score, 3),
            is_ambiguous=False
        )

    def process_tracks(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        """
        Evalúa reconocimiento facial sobre tracks en ATTENTION_CANDIDATE o ENGAGING
        que presentan orientación frontal razonable (incluso a distancia).
        """
        now = time.monotonic()
        for track in tracks:
            # Omitir tracks que dejaron de verse (consecutive_misses > 0)
            if track.consecutive_misses > 0:
                continue

            # Si el track está en DISENGAGED, preservar su identidad conocida sin resetearla
            if track.state == TrackState.DISENGAGED:
                continue

            # Evaluar en CANDIDATE, ENGAGING o en OBSERVED con orientación frontal preliminar
            if track.state not in (TrackState.ENGAGING, TrackState.ATTENTION_CANDIDATE):
                if track.state == TrackState.OBSERVED and track.dwell_time >= 0.20 and track.orientation_frontal_score >= 0.55:
                    pass
                else:
                    if track.identity_decision not in ("KNOWN", "UNKNOWN"):
                        track.identity_decision = "NOT_EVALUATED"
                        track.identity_label = "NOT_EVALUATED"
                    continue

            # Verificación periódica continua de tracks KNOWN (cada 1.5s):
            # Previene que un transeúnte o persona cercana retenga la identidad si el tracker saltó erróneamente
            if track.identity_decision == "KNOWN" and track.recognition_eval_count >= self.config.samples_for_consolidation:
                if (now - getattr(track, '_last_identity_verification_time', 0.0)) >= 1.5:
                    if track.raw_detection is not None and getattr(track, 'is_complete_face', True) and track.orientation_frontal_score >= 0.55:
                        track._last_identity_verification_time = now
                        try:
                            v_aligned = self.recognizer.align_crop(frame, track.raw_detection)
                            v_feat = self.recognizer.extract_feature(v_aligned)
                            v_match = self.match_embedding(v_feat)
                            is_still_same = (
                                v_match.decision == "KNOWN" and
                                v_match.label.strip().lower() == track.identity_label.strip().lower() and
                                v_match.similarity >= (self.config.similarity_threshold - 0.06)
                            )
                            if not is_still_same:
                                track._identity_mismatch_count = getattr(track, '_identity_mismatch_count', 0) + 1
                                logger.warning(
                                    f"[IDENTITY-CHECK-WARN] Track #{track.track_id} ('{track.identity_label}') "
                                    f"dio discordancia biométrica (match='{v_match.label}', sim={v_match.similarity:.2f}, "
                                    f"fallos={track._identity_mismatch_count}/2)"
                                )
                                if track._identity_mismatch_count >= 2:
                                    logger.warning(
                                        f"[IDENTITY-RESET] Revocando identidad de Track #{track.track_id} "
                                        f"('{track.identity_label}') por falta de concordancia biométrica continua."
                                    )
                                    track.identity_decision = "NOT_EVALUATED"
                                    track.identity_label = "NOT_EVALUATED"
                                    track.person_id = None
                                    track.recognition_eval_count = 0
                                    track.recognition_candidate_history.clear()
                                    track.embedding_samples.clear()
                                    track._identity_mismatch_count = 0
                            else:
                                track._identity_mismatch_count = 0
                                track.recognition_similarity = v_match.similarity
                                self._maybe_adaptive_update(track, v_feat, v_aligned)
                        except Exception as e:
                            logger.debug(f"Error en verificación periódica de Track #{track.track_id}: {e}")
                continue

            # GUARD: Evaluar si el track es apto para muestreo biométrico.
            # Requiere que el rostro esté COMPLETO (no cortado en bordes ni ocluido).
            # Si el rostro está parcialmente visible (borde de cámara, inclinación extrema),
            # NO se muestrea NI se avanza el contador de evaluación. El track permanece
            # en su estado actual (NOT_EVALUATED o IDENTIFYING) sin consolidar como UNKNOWN.
            # Esto previene categorizar a personas registradas como 'Visitante' cuando
            # solo se ve un fragmento del rostro.
            if not getattr(track, 'is_complete_face', True):
                # Si nunca se ha muestreado, mantenerlo explícitamente como NOT_EVALUATED
                if track.recognition_eval_count == 0 and track.identity_decision not in ("KNOWN",):
                    track.identity_decision = "NOT_EVALUATED"
                    track.identity_label = "NOT_EVALUATED"
                continue

            last_sample_t = getattr(track, '_last_recognition_sample_time', 0.0)

            allow_eval = False
            if track.recognition_eval_count < self.config.samples_for_consolidation:
                if track.raw_detection is not None and track.orientation_frontal_score >= 0.55 and track.height >= 45:
                    if (now - last_sample_t) >= 0.06:  # Muestreo ágil cada ~60 ms (~2 frames a 30 FPS)
                        allow_eval = True
            elif track.identity_decision == "UNKNOWN" and track.raw_detection is not None and track.orientation_frontal_score >= 0.55:
                if (now - last_sample_t) >= 0.08:
                    allow_eval = True

            if not allow_eval:
                continue

            track._last_recognition_sample_time = now

            # Extraer embedding del rostro alineado
            try:
                aligned = self.recognizer.align_crop(frame, track.raw_detection)
                feat = self.recognizer.extract_feature(aligned)
                match = self.match_embedding(feat)

                feat_flat = feat.flatten().astype(np.float32)

                if track.recognition_eval_count < self.config.samples_for_consolidation:
                    track.embedding_samples.append(feat_flat)
                    track.embedding = feat_flat

                    track.recognition_candidate_history.append((match.decision, match.label, match.similarity))
                    track.recognition_eval_count += 1
                    track.recognition_similarity = match.similarity

                    # Si el primer o segundo match es KNOWN con buena confianza, asignar de inmediato
                    # para evitar que parpadee como 'UNKNOWN' o 'Visitante'
                    if match.decision == "KNOWN":
                        track.identity_label = match.label
                        track.person_id = match.person_id
                        if match.similarity >= (self.config.similarity_threshold + 0.04):
                            # Coincidencia sólida: consolidar de inmediato
                            track.identity_decision = "KNOWN"
                            logger.info(
                                f"[FAST-MATCH] Track #{track.track_id} identificado de inmediato como '{match.label}' "
                                f"(Sim: {match.similarity:.2f})"
                            )
                        else:
                            track.identity_decision = "IDENTIFYING"
                    else:
                        track.identity_decision = "IDENTIFYING"

                    # Verificar si alcanzamos el número de muestras para consolidación formal
                    if track.recognition_eval_count >= self.config.samples_for_consolidation and track.identity_decision != "KNOWN":
                        self._consolidate_track_identity(track)
                else:
                    # Muestra de rescate para track previamente UNKNOWN
                    if match.decision == "KNOWN":
                        logger.info(
                            f"[DIAGNOSTICS] Track #{track.track_id} rescatado de UNKNOWN a KNOWN: "
                            f"'{match.label}' (Sim: {match.similarity:.2f}, Orient: {track.orientation_frontal_score:.2f})"
                        )
                        track.identity_decision = "KNOWN"
                        track.identity_label = match.label
                        track.person_id = match.person_id
                        track.recognition_similarity = match.similarity
                        track.embedding = feat_flat
                    else:
                        # Si sigue siendo UNKNOWN y no tiene visitor_id, resolverlo
                        if getattr(track, 'visitor_id', None) is None:
                            self.resolve_visitor_identity(track)

                # Enrolamiento adaptativo en segundo plano si la identidad es KNOWN
                if track.identity_decision == "KNOWN" and track.person_id:
                    self._maybe_adaptive_update(track, feat, aligned)

            except Exception as e:
                logger.error(f"Error procesando reconocimiento para Track #{track.track_id}: {e}")

        return tracks

    def _maybe_adaptive_update(self, track: Track, feat: np.ndarray, crop: Optional[np.ndarray] = None):
        """
        Enrolamiento adaptativo pasivo en segundo plano:
        Si una persona consolidada como KNOWN es vista con muy alta confianza (>= 0.58) y
        orientación frontal sólida, actualiza paulatinamente su catálogo biométrico.
        Permite que el sistema absorba cambios graduales (crecimiento de barba, etc.) sin fotos manuales.
        """
        if not getattr(self.config, 'adaptive_learning_enabled', True):
            return
        if not track.person_id or track.identity_decision != "KNOWN":
            return
        if track.recognition_similarity < 0.58:
            return
        if track.orientation_frontal_score < 0.65 or track.height < 75:
            return

        now = time.monotonic()
        last_update = self._last_adaptive_update.get(track.person_id, 0.0)
        cooldown = getattr(self.config, 'adaptive_cooldown_seconds', 1800.0)

        if (now - last_update) < cooldown:
            return

        self._last_adaptive_update[track.person_id] = now

        # Ejecutar persistencia SQLite y recarga de catálogo en segundo plano
        # para evitar micro-congelamientos de ~200ms en el hilo principal de video
        feat_copy = feat.copy()
        crop_copy = crop.copy() if crop is not None else None
        person_id = track.person_id
        identity_label = track.identity_label
        similarity = track.recognition_similarity
        orientation = track.orientation_frontal_score

        def _bg_update_worker():
            try:
                thumb_bytes = None
                if crop_copy is not None:
                    ret, buf = cv2.imencode(".jpg", crop_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        thumb_bytes = buf.tobytes()

                self.identity_store.add_embedding_to_person(
                    person_id=person_id,
                    embedding=feat_copy,
                    source_type="adaptive_auto",
                    thumbnail_bytes=thumb_bytes
                )
                # Conservar catálogo limpio y eficiente (máx 4 adaptativas)
                self.identity_store.prune_adaptive_embeddings(person_id, max_samples=4)
                self.reload_identities()
                logger.info(
                    f"[ADAPTIVE] Auto-actualizado perfil biométrico de '{identity_label}' "
                    f"(Similitud: {similarity:.2f}, Orientación: {orientation:.2f})"
                )
            except Exception as e:
                logger.error(f"Error en auto-actualización adaptativa para '{identity_label}': {e}")

        threading.Thread(
            target=_bg_update_worker,
            daemon=True,
            name=f"adaptive-update-{person_id}"
        ).start()

    def resolve_visitor_identity(self, track: Track) -> int:
        """
        Asigna o reasigna una identidad de visitante consistente (Visitante #N)
        basándose en similitud de embedding facial contra visitantes recientes.
        Evita que la misma persona cambie de 'Visitante #1' a 'Visitante #2' al salir y volver.
        """
        now = time.monotonic()
        # 1. Purgar visitantes expirados
        self._recent_visitors = [
            v for v in self._recent_visitors
            if (now - v["last_seen"]) < self._visitor_memory_window_seconds
        ]

        # 2. Si el track ya tiene un visitor_id asignado y está en memoria, actualizar y retornar
        if track.visitor_id is not None:
            for v in self._recent_visitors:
                if v["visitor_id"] == track.visitor_id:
                    v["last_seen"] = now
                    if track.embedding is not None:
                        emb_norm = track.embedding / max(1e-6, np.linalg.norm(track.embedding))
                        v["embedding"] = 0.85 * v["embedding"] + 0.15 * emb_norm
                        v["embedding"] /= max(1e-6, np.linalg.norm(v["embedding"]))
                    return track.visitor_id

        # 3. Si tenemos embedding, buscar coincidencia con visitantes recientes
        if track.embedding is not None and len(self._recent_visitors) > 0:
            q_emb = np.ascontiguousarray(track.embedding, dtype=np.float32).flatten()
            q_norm = float(np.linalg.norm(q_emb))
            if q_norm > 1e-6:
                best_sim = -1.0
                best_visitor = None
                for v in self._recent_visitors:
                    ref_emb = v["embedding"]
                    ref_norm = float(np.linalg.norm(ref_emb))
                    if ref_norm > 1e-6:
                        sim = float(np.dot(q_emb, ref_emb) / (q_norm * ref_norm))
                        if sim > best_sim:
                            best_sim = sim
                            best_visitor = v

                # Umbral de reidentificación de visitante (0.58)
                if best_sim >= 0.58 and best_visitor is not None:
                    v_id = best_visitor["visitor_id"]
                    best_visitor["last_seen"] = now
                    emb_norm = q_emb / q_norm
                    best_visitor["embedding"] = 0.85 * best_visitor["embedding"] + 0.15 * emb_norm
                    best_visitor["embedding"] /= max(1e-6, np.linalg.norm(best_visitor["embedding"]))

                    track.visitor_id = v_id
                    track.identity_label = f"Visitante #{v_id}"
                    logger.info(
                        f"[VISITOR-REID] Track #{track.track_id} re-identificado como Visitante #{v_id} "
                        f"(Sim: {best_sim:.2f})"
                    )
                    return v_id

        # 4. Si no coincidió con ningún visitante reciente, asignar nuevo ID secuencial
        v_id = self._next_visitor_id
        self._next_visitor_id += 1
        track.visitor_id = v_id
        track.identity_label = f"Visitante #{v_id}"

        if track.embedding is not None:
            emb_norm = track.embedding / max(1e-6, np.linalg.norm(track.embedding))
            self._recent_visitors.append({
                "visitor_id": v_id,
                "embedding": emb_norm.astype(np.float32),
                "first_seen": now,
                "last_seen": now
            })
        logger.info(f"[NEW-VISITOR] Track #{track.track_id} asignado como nuevo Visitante #{v_id}")
        return v_id

    def _consolidate_track_identity(self, track: Track):
        """
        Consolida la identidad final tras recolectar N muestras independientes.
        """
        if track.embedding_samples:
            mean_emb = np.mean(track.embedding_samples, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-6:
                mean_emb = mean_emb / norm
            track.embedding = mean_emb.astype(np.float32)

        history = track.recognition_candidate_history
        known_votes: Dict[str, List[float]] = {}
        unknown_votes = 0

        for decision, label, sim in history:
            if decision == "KNOWN":
                known_votes.setdefault(label, []).append(sim)
            else:
                unknown_votes += 1

        # Encontrar candidato más votado
        if known_votes:
            best_cand, sims = max(known_votes.items(), key=lambda item: len(item[1]))
            if len(sims) >= max(1, int(self.config.samples_for_consolidation * 0.4)):
                track.identity_decision = "KNOWN"
                track.identity_label = best_cand
                track.recognition_similarity = round(float(np.mean(sims)), 3)
                logger.info(
                    f"[DIAGNOSTICS] Identidad consolidada para Track #{track.track_id}: "
                    f"'{best_cand}' (Sim: {track.recognition_similarity:.2f}, votos: {len(sims)}/{len(history)})"
                )
                return

        # Si no hubo consenso claro o dominó UNKNOWN
        track.identity_decision = "UNKNOWN"
        all_sims = [sim for _, _, sim in history]
        track.recognition_similarity = round(float(max(all_sims)), 3) if all_sims else 0.0
        self.resolve_visitor_identity(track)
        logger.info(f"[DIAGNOSTICS] Identidad consolidada para Track #{track.track_id} en {time.monotonic():.3f}: {track.identity_label} (Similitud máx: {track.recognition_similarity:.2f})")
