"""Step 5: QC verification with BlazeFace detector."""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class BlazeFaceQc:
    def __init__(self, model_path, min_score=0.2):
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_score,
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self.min_score = min_score

    def detect_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(image)

        scores = []
        if not result.detections:
            return scores

        for detection in result.detections:
            if detection.categories:
                scores.append(float(detection.categories[0].score))
        return scores

    def close(self):
        self.detector.close()
