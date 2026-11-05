"""Step 3: Simple IOU face tracker."""


def iou(a, b):
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = a.w * a.h
    area_b = b.w * b.h
    return inter / float(area_a + area_b - inter)


class Track:
    def __init__(self, box):
        self.box = box
        self.age = 0


class SimpleIouTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, detections):
        matched_det = set()
        matched_trk = set()

        for ti, track in enumerate(self.tracks):
            best_iou = 0.0
            best_di = -1

            for di, det in enumerate(detections):
                if di in matched_det:
                    continue
                score = iou(track.box, det)
                if score > best_iou:
                    best_iou = score
                    best_di = di

            if best_di >= 0 and best_iou >= self.iou_threshold:
                self.tracks[ti].box = detections[best_di]
                self.tracks[ti].age = 0
                matched_det.add(best_di)
                matched_trk.add(ti)

        for di, det in enumerate(detections):
            if di not in matched_det:
                self.tracks.append(Track(det))

        surviving = []
        for ti, track in enumerate(self.tracks):
            if ti in matched_trk:
                surviving.append(track)
                continue
            track.age += 1
            if track.age <= self.max_age:
                surviving.append(track)

        self.tracks = surviving
        return [track.box for track in self.tracks]
