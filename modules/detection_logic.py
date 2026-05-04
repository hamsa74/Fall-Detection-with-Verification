import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import urllib.request
import os

MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[Setup] Downloading pose model... please wait")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Setup] Model downloaded successfully!")

# Unique color per person (BGR)
PERSON_COLORS = [
    (80,  200, 120),   # Person 1 - Green
    (60,  140, 220),   # Person 2 - Blue
    (60,  60,  220),   # Person 3 - Red
    (0,   200, 220),   # Person 4 - Yellow
]

class PersonTracker:
    """
    Multi-person tracker using mediapipe Tasks API.
    Tracks up to `max_persons` people, each with a stable ID and color.
    Uses IoU-based matching to keep IDs consistent across frames.
    """

    def __init__(self, max_persons=3, smooth_window=5, visibility_threshold=0.5):
        ensure_model()

        self.max_persons       = max_persons
        self.visibility_threshold = visibility_threshold
        self.frame_index       = 0

        # Per-person smoothing histories  {person_id: deque}
        self.box_histories = {}
        self.smooth_window = smooth_window

        # Last known boxes for ID matching  {person_id: box}
        self.last_boxes = {}
        self.next_id    = 0

        # Try GPU (CUDA) first, fall back to CPU if unavailable
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions as BO
            gpu_delegate = python.BaseOptions.Delegate.GPU
            base_options = python.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=gpu_delegate
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=max_persons,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose_analyzer = vision.PoseLandmarker.create_from_options(options)
            print("[Detection] ✅ Running on GPU (CUDA)")
        except Exception as e:
            print(f"[Detection] ⚠ GPU unavailable ({e}), falling back to CPU")
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=max_persons,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose_analyzer = vision.PoseLandmarker.create_from_options(options)
            print("[Detection] Running on CPU")

    def _filter_landmarks(self, pts, w, h):
        vx, vy = [], []
        for p in pts:
            if p.visibility >= self.visibility_threshold:
                vx.append(p.x)
                vy.append(p.y)
        return vx, vy

    def _landmarks_to_box(self, pts, w, h):
        vx, vy = self._filter_landmarks(pts, w, h)
        if len(vx) < 5:
            return None
        sx, ex = int(min(vx)*w), int(max(vx)*w)
        sy, ey = int(min(vy)*h), int(max(vy)*h)
        px, py = int(0.05*w), int(0.05*h)
        return [max(0,sx-px), max(0,sy-py),
                min(w, ex-sx+2*px), min(h, ey-sy+2*py)]

    def _iou(self, b1, b2):
        """Intersection over Union for two boxes [x,y,w,h]."""
        x1,y1,w1,h1 = b1;  x2,y2,w2,h2 = b2
        ix = max(0, min(x1+w1,x2+w2) - max(x1,x2))
        iy = max(0, min(y1+h1,y2+h2) - max(y1,y2))
        inter = ix * iy
        union = w1*h1 + w2*h2 - inter
        return inter/union if union > 0 else 0

    def _match_ids(self, raw_boxes):
        """
        Match detected boxes to existing IDs using IoU.
        New person → new ID. Disappeared person → ID removed.
        """
        matched   = {}   # {person_id: box}
        used_raw  = set()

        # Try to match each known ID to a detected box
        for pid, last_box in self.last_boxes.items():
            best_iou, best_i = 0, -1
            for i, rb in enumerate(raw_boxes):
                if i in used_raw:
                    continue
                iou = self._iou(last_box, rb)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_iou > 0.15 and best_i >= 0:
                matched[pid] = raw_boxes[best_i]
                used_raw.add(best_i)

        # New detections that didn't match anyone → assign new IDs
        for i, rb in enumerate(raw_boxes):
            if i not in used_raw:
                matched[self.next_id] = rb
                self.next_id += 1

        self.last_boxes = matched
        return matched

    def _smooth_box(self, pid, box):
        if pid not in self.box_histories:
            self.box_histories[pid] = deque(maxlen=self.smooth_window)
        hist = self.box_histories[pid]
        hist.append(box)
        return [int(sum(b[i] for b in hist)/len(hist)) for i in range(4)]

    def get_persons(self, frame):
        """
        Returns list of dicts:
          { 'id': int, 'box': [x,y,w,h], 'landmarks': [...], 'color': (B,G,R) }
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = self.frame_index * 33
        self.frame_index += 1
        result = self.pose_analyzer.detect_for_video(mp_img, ts)

        if not result.pose_landmarks:
            self.last_boxes = {}
            self.box_histories = {}
            return []

        # Build raw boxes for all detected poses
        raw_boxes = []
        raw_lms   = []
        for lms in result.pose_landmarks:
            box = self._landmarks_to_box(lms, w, h)
            if box:
                raw_boxes.append(box)
                raw_lms.append(lms)

        # Match to stable IDs
        id_to_box = self._match_ids(raw_boxes)

        persons = []
        for pid, box in id_to_box.items():
            # Find landmarks for this box
            best_lms = None
            best_iou = 0
            for lms, rb in zip(raw_lms, raw_boxes):
                iou = self._iou(box, rb)
                if iou > best_iou:
                    best_iou, best_lms = iou, lms

            if best_lms is None:
                continue

            smooth = self._smooth_box(pid, box)
            color  = PERSON_COLORS[pid % len(PERSON_COLORS)]

            persons.append({
                'id':        pid,
                'box':       smooth,
                'landmarks': best_lms,
                'color':     color,
            })

        return persons