"""
Módulo de almacenamiento y persistencia de identidades locales (SQLite).
Garantiza:
- Privacidad absoluta: 100% offline, sin transmisión a la nube ni almacenamiento de fotos completas.
- Estructura 1 a N: Una persona (UUID único) puede tener múltiples muestras de embeddings.
- Almacenamiento eficiente: Vector de 128 floats32 serializado en BLOB (512 bytes).
"""

import os
import sqlite3
import uuid
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from contextlib import contextmanager

from audacia.app.core.paths import resolve_path

logger = logging.getLogger(__name__)


class IdentityStore:
    """
    Gestiona el catálogo de identidades registradas en SQLite local.
    """

    def __init__(self, db_path: str = "data/identities.db"):
        self.db_path = resolve_path(db_path)
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()
        logger.info(f"IdentityStore conectado a base de datos SQLite: '{db_path}'")

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        """Crea las tablas persons y person_embeddings si no existen."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                notes TEXT
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS person_embeddings (
                embedding_id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_type TEXT NOT NULL,
                thumbnail BLOB,
                created_at REAL NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
            );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_person_id ON person_embeddings(person_id);")

    def create_person(
        self,
        name: str,
        embedding: np.ndarray,
        source_type: str = "photo",
        thumbnail_bytes: Optional[bytes] = None,
        notes: str = ""
    ) -> str:
        """
        Registra una nueva persona con su embedding inicial de 128-D.
        Retorna el person_id (UUID generado).
        """
        name_clean = name.strip()
        if not name_clean:
            raise ValueError("El nombre de la persona no puede estar vacío.")

        person_id = str(uuid.uuid4())
        emb_id = str(uuid.uuid4())
        now = time.monotonic()

        # Serializar vector float32 a bytes binarios (512 bytes exactos)
        emb_arr = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, 128)
        emb_bytes = emb_arr.tobytes()

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO persons (person_id, name, created_at, updated_at, notes) VALUES (?, ?, ?, ?, ?);",
                (person_id, name_clean, now, now, notes)
            )
            conn.execute(
                "INSERT INTO person_embeddings (embedding_id, person_id, embedding, source_type, thumbnail, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                (emb_id, person_id, emb_bytes, source_type, thumbnail_bytes, now)
            )

        logger.info(f"Persona '{name_clean}' registrada con ID: {person_id}")
        return person_id

    def add_embedding_to_person(
        self,
        person_id: str,
        embedding: np.ndarray,
        source_type: str = "camera",
        thumbnail_bytes: Optional[bytes] = None
    ) -> str:
        """Agrega un embedding adicional a una persona existente (1 a N)."""
        emb_id = str(uuid.uuid4())
        now = time.monotonic()
        emb_arr = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, 128)
        emb_bytes = emb_arr.tobytes()

        with self._get_connection() as conn:
            # Verificar existencia
            row = conn.execute("SELECT person_id FROM persons WHERE person_id = ?;", (person_id,)).fetchone()
            if not row:
                raise KeyError(f"No existe ninguna persona con ID: '{person_id}'.")

            conn.execute(
                "INSERT INTO person_embeddings (embedding_id, person_id, embedding, source_type, thumbnail, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                (emb_id, person_id, emb_bytes, source_type, thumbnail_bytes, now)
            )
            conn.execute("UPDATE persons SET updated_at = ? WHERE person_id = ?;", (now, person_id))

        return emb_id

    def get_embeddings_by_person_id(self, person_id: str) -> List[np.ndarray]:
        """Obtiene todos los vectores de características 128-D asociados a un person_id."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT embedding FROM person_embeddings WHERE person_id = ?;",
                (person_id,)
            ).fetchall()
            return [np.frombuffer(r["embedding"], dtype=np.float32).reshape(1, 128) for r in rows]

    def prune_adaptive_embeddings(self, person_id: str, max_samples: int = 4):
        """Mantiene un límite saludable de muestras auto-adaptativas para no saturar memoria."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT embedding_id FROM person_embeddings WHERE person_id = ? AND source_type LIKE 'adaptive%' ORDER BY created_at ASC;",
                (person_id,)
            ).fetchall()
            if len(rows) > max_samples:
                excess = len(rows) - max_samples
                for r in rows[:excess]:
                    conn.execute("DELETE FROM person_embeddings WHERE embedding_id = ?;", (r["embedding_id"],))

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """
        Carga todos los embeddings registrados en memoria para comparación rápida.
        Retorna lista de dicts: [{'person_id', 'name', 'embedding': np.ndarray, 'source_type'}]
        """
        results = []
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT p.person_id, p.name, e.embedding_id, e.embedding, e.source_type
            FROM persons p
            JOIN person_embeddings e ON p.person_id = e.person_id;
            """).fetchall()

            for r in rows:
                feat = np.frombuffer(r["embedding"], dtype=np.float32).reshape(1, 128)
                results.append({
                    "person_id": r["person_id"],
                    "name": r["name"],
                    "embedding_id": r["embedding_id"],
                    "embedding": feat,
                    "source_type": r["source_type"]
                })
        return results

    def list_persons(self) -> List[Dict[str, Any]]:
        """Lista todas las identidades registradas con su número de muestras."""
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT p.person_id, p.name, p.created_at, p.updated_at, p.notes, COUNT(e.embedding_id) as sample_count
            FROM persons p
            LEFT JOIN person_embeddings e ON p.person_id = e.person_id
            GROUP BY p.person_id
            ORDER BY p.name ASC;
            """).fetchall()

            return [dict(r) for r in rows]

    def delete_person(self, person_id: str) -> bool:
        """Elimina una persona y todos sus embeddings asociados."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM persons WHERE person_id = ?;", (person_id,))
            return cur.rowcount > 0

    def find_person_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Busca persona por nombre exacto."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM persons WHERE LOWER(name) = LOWER(?);", (name.strip(),)).fetchone()
            return dict(row) if row else None
