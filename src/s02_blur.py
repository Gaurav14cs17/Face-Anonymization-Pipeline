"""Step 2: Blur detected face regions."""

import cv2

from s02_detection import FaceBox


class FaceBlurrer:
    def __init__(self, kernel=51, draw_boxes=False, expand_scale=0.15):
        self.kernel = kernel if kernel % 2 == 1 else kernel + 1
        self.draw_boxes = draw_boxes
        self.expand_scale = expand_scale

    def apply(self, frame, boxes):
        if not boxes:
            return frame

        out = frame.copy()
        frame_h, frame_w = out.shape[:2]

        for box in boxes:
            region = self._expand(box, frame_w, frame_h)
            x1, y1, x2, y2 = region.as_xyxy()
            roi = out[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            blurred = cv2.GaussianBlur(roi, (self.kernel, self.kernel), 0)
            out[y1:y2, x1:x2] = blurred

            if self.draw_boxes:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return out

    def _expand(self, box, frame_w, frame_h):
        pad_w = int(box.w * self.expand_scale)
        pad_h = int(box.h * self.expand_scale)
        x = max(0, box.x - pad_w)
        y = max(0, box.y - pad_h)
        w = min(frame_w - x, box.w + 2 * pad_w)
        h = min(frame_h - y, box.h + 2 * pad_h)
        return FaceBox(x, y, w, h, box.score)
