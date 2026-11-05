"""Steps 2-3: Face anonymization pipeline (detection + blur + tracking)."""

import cv2

from s02_detection import YuNetDetector
from s02_blur import FaceBlurrer
from s03_tracker import SimpleIouTracker
from s05_verify import BlazeFaceQc


class FaceAnonymizer:
    def __init__(self, yunet_model, qc_model, draw_boxes=False, use_tracker=True):
        self.detector = YuNetDetector(yunet_model)
        self.tracker = SimpleIouTracker() if use_tracker else None
        self.blurrer = FaceBlurrer(draw_boxes=draw_boxes)
        self.qc_model = qc_model

    def process(self, input_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.tracker is not None:
            self.tracker.reset()

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return {
                "input": str(input_path),
                "output": str(output_path),
                "input_frame_count": 0,
                "output_frame_count": 0,
                "needs_review": True,
                "qc_face_detections": 0,
                "qc_max_score": 0.0,
                "qc_notes": ["input_decode_failed"],
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        input_frames = 0
        output_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            input_frames += 1
            boxes = self.detector.detect(frame)

            if self.tracker is not None:
                boxes = self.tracker.update(boxes)

            writer.write(self.blurrer.apply(frame, boxes))
            output_frames += 1

        cap.release()
        writer.release()

        qc = self.check_output(output_path, input_frames, output_frames)
        return {
            "input": str(input_path),
            "output": str(output_path),
            "input_frame_count": input_frames,
            "output_frame_count": output_frames,
            "needs_review": qc["needs_review"],
            "qc_face_detections": qc["face_detections"],
            "qc_max_score": qc["max_score"],
            "qc_notes": qc["notes"],
        }

    def check_output(self, output_path, input_frames, output_frames):
        notes = []

        if input_frames != output_frames:
            notes.append(
                "frame_count_mismatch:%s!=%s" % (input_frames, output_frames)
            )

        qc = BlazeFaceQc(self.qc_model)
        cap = cv2.VideoCapture(str(output_path))

        if not cap.isOpened():
            qc.close()
            notes.append("output_decode_failed")
            return {
                "needs_review": True,
                "face_detections": 0,
                "max_score": 0.0,
                "notes": notes,
            }

        total_faces = 0
        max_score = 0.0
        index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if index % 5 == 0:
                for score in qc.detect_frame(frame):
                    total_faces += 1
                    max_score = max(max_score, score)
                    if score >= 0.2:
                        notes.append(
                            "residual_face_score:%.3f@frame:%s" % (score, index)
                        )

            index += 1

        cap.release()
        qc.close()

        return {
            "needs_review": bool(notes) or max_score >= 0.2,
            "face_detections": total_faces,
            "max_score": max_score,
            "notes": notes,
        }
