"""Step 4: Job storage with SQLite and SHA-256 idempotency."""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class JobRecord:
    def __init__(
        self,
        job_id,
        content_hash,
        status,
        input_path,
        output_path,
        needs_review,
        qc_notes,
        created_at,
        updated_at,
    ):
        self.job_id = job_id
        self.content_hash = content_hash
        self.status = status
        self.input_path = input_path
        self.output_path = output_path
        self.needs_review = needs_review
        self.qc_notes = qc_notes
        self.created_at = created_at
        self.updated_at = updated_at


class JobStore:
    def __init__(self, db_path):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    content_hash TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    output_path TEXT,
                    needs_review INTEGER,
                    qc_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def hash_bytes(data):
        return hashlib.sha256(data).hexdigest()

    def get_by_hash(self, content_hash):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def get(self, job_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def create_pending(self, content_hash, input_path):
        job_id = str(uuid.uuid4())
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, content_hash, status, input_path, output_path,
                    needs_review, qc_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                """,
                (job_id, content_hash, "pending", str(input_path), "[]", now, now),
            )
        return JobRecord(
            job_id=job_id,
            content_hash=content_hash,
            status="pending",
            input_path=str(input_path),
            output_path=None,
            needs_review=None,
            qc_notes=[],
            created_at=now,
            updated_at=now,
        )

    def mark_complete(self, job_id, output_path, needs_review, qc_notes):
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, output_path = ?, needs_review = ?,
                qc_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    "completed",
                    str(output_path),
                    1 if needs_review else 0,
                    json.dumps(qc_notes),
                    now,
                    job_id,
                ),
            )

    def mark_failed(self, job_id, note):
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, qc_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                ("failed", json.dumps([note]), now, job_id),
            )

    def _row_to_job(self, row):
        qc_raw = row["qc_json"] or "[]"
        notes = json.loads(qc_raw)
        needs = row["needs_review"]
        return JobRecord(
            job_id=row["job_id"],
            content_hash=row["content_hash"],
            status=row["status"],
            input_path=row["input_path"],
            output_path=row["output_path"],
            needs_review=bool(needs) if needs is not None else None,
            qc_notes=notes if isinstance(notes, list) else [],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def to_public_dict(self, job):
        return {
            "job_id": job.job_id,
            "content_hash": job.content_hash,
            "status": job.status,
            "output_path": job.output_path,
            "needs_review": job.needs_review,
            "qc_notes": job.qc_notes,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
