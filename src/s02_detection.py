"""Step 2: YuNet face detection."""

import cv2


class FaceBox:
    def __init__(self, x, y, w, h, score):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score

    def as_xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def clip(self, width, height):
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(width, self.x + self.w)
        y2 = min(height, self.y + self.h)
        return FaceBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1), self.score)


class YuNetDetector:
    def __init__(self, model_path, score_threshold=0.6):
        self.model_path = str(model_path)
        self.score_threshold = score_threshold
        self.detector = None
        self.input_size = None

    def detect(self, frame):
        height, width = frame.shape[:2]
        
        if self.detector is None:
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (width, height),
                self.score_threshold,
                0.3,
                5000,
            )
            self.input_size = (width, height)
        elif self.input_size != (width, height):
            self.detector.setInputSize((width, height))
            self.input_size = (width, height)

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        boxes = []
        for face in faces:
            x, y, w, h = face[:4]
            score = float(face[-1])
            box = FaceBox(int(x), int(y), int(w), int(h), score).clip(width, height)
            if box.w > 0 and box.h > 0:
                boxes.append(box)
        return boxes
