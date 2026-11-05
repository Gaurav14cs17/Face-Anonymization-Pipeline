"""Step 4: Idempotent async API with job persistence."""

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from s23_anonymizer import FaceAnonymizer
from s04_jobs import JobStore

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
UPLOAD_DIR = ROOT / "output" / "uploads"
OUTPUT_DIR = ROOT / "output" / "anonymized"
DB_PATH = ROOT / "jobs.db"

YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
BLAZE = MODELS / "blaze_face_short_range.tflite"

store = JobStore(DB_PATH)
app = FastAPI(title="Stage01 Naive Anonymizer", version="0.1.0")


def _process_job(job_id, input_path, content_hash):
    out_path = OUTPUT_DIR / (content_hash[:16] + "_anonymized.mp4")
    try:
        anonymizer = FaceAnonymizer(YUNET, BLAZE)
        result = anonymizer.process(input_path, out_path)
        store.mark_complete(
            job_id,
            output_path=out_path,
            needs_review=result["needs_review"],
            qc_notes=result["qc_notes"],
        )
    except Exception as exc:
        store.mark_failed(job_id, str(exc))


@app.post("/v1/anonymize")
async def anonymize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    data = await file.read()
    content_hash = store.hash_bytes(data)
    existing = store.get_by_hash(content_hash)
    if existing:
        return {"idempotent_reuse": True, **store.to_public_dict(existing)}

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    input_path = UPLOAD_DIR / (content_hash[:16] + suffix)
    input_path.write_bytes(data)

    job = store.create_pending(content_hash, input_path)
    background_tasks.add_task(_process_job, job.job_id, input_path, content_hash)
    return {"idempotent_reuse": False, **store.to_public_dict(job)}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id):
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return store.to_public_dict(job)
