# Face Anonymization Pipeline

Automatically detect and blur every face in a video using **YuNet** detection, IoU
tracking, and **BlazeFace** quality-check verification.

## Before / After

| Raw input | Anonymized output |
| :---: | :---: |
| [![Raw input](assets/raw_preview.gif)](assets/raw.mp4) | [![Anonymized output](assets/anonymized_preview.gif)](assets/anonymized.mp4) |

> Animated previews (first 8 seconds). Click either preview to open the full video:
> [raw](assets/raw.mp4) · [anonymized](assets/anonymized.mp4)

## Quick Start

```bash
# Setup
cd /home/ggoswami/project_o1/stage01_naive
source ../.venv/bin/activate
pip install -r requirements.txt

# Blur faces in a video
python scripts/run_local.py data/samples/youtube_-Koj9hvcBMk.mp4

# Result goes to: output/youtube_-Koj9hvcBMk_anonymized.mp4
```

## See detected faces (for testing)

```bash
python scripts/run_local.py data/samples/youtube_-Koj9hvcBMk.mp4 --draw-boxes
```

Green boxes show where faces were detected and blurred.

## Run API server

```bash
cd src
uvicorn s04_app:app --reload
```

Upload videos: `POST /v1/anonymize`
Check status: `GET /v1/jobs/{job_id}`

## Code Structure

```
src/
├── s02_detection.py    - Step 2: Face detection (YuNet)
├── s02_blur.py         - Step 2: Blur faces
├── s03_tracker.py      - Step 3: Track faces
├── s04_jobs.py         - Step 4: Job storage
├── s04_app.py          - Step 4: API server
├── s05_verify.py       - Step 5: QC verification
└── s23_anonymizer.py   - Steps 2-3: Main pipeline
```

## Models

- **YuNet** (anonymization): `models/face_detection_yunet_2023mar.onnx`
- **BlazeFace** (quality check): `models/blaze_face_short_range.tflite`

## Sample Output

```json
{
  "input": "input.mp4",
  "output": "output/input_anonymized.mp4",
  "input_frame_count": 3230,
  "output_frame_count": 3230,
  "needs_review": true,
  "qc_face_detections": 716,
  "qc_max_score": 0.906
}
```

## More Details

See `docs/` folder for step-by-step explanations and `RESULTS_SUMMARY.md` for
detection/QC metrics and known failure modes.
