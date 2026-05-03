import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import urllib.request
import os

# Download the pose landmarker model if not present
MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[Setup] Downloading pose model... please wait")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Setup] Model downloaded successfully!")

class PersonTracker:
    """
    PersonTracker using the NEW mediapipe Tasks API (>= 0.10.30).
    - PoseLandmarker instead of deprecated mp.solutions.pose
    - Bounding box smoothing
    - Visibility filtering
    """

    def __init__(self, smooth_window=5, visibility_threshold=0.5):
        ensure_model()

        self.visibility_threshold = visibility_threshold
        self.box_history = deque(maxlen=smooth_window)
        self.latest_landmarks = None

        # New Tasks API setup
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_analyzer = vision.PoseLandmarker.create_from_options(options)
        self.frame_index = 0

    def _filter_landmarks(self, pts):
        visible_x, visible_y = [], []
        for p in pts:
            if p.visibility >= self.visibility_threshold:
                visible_x.append(p.x)
                visible_y.append(p.y)
        return visible_x, visible_y

    def _smooth_box(self, box):
        self.box_history.append(box)
        avg_x = int(sum(b[0] for b in self.box_history) / len(self.box_history))
        avg_y = int(sum(b[1] for b in self.box_history) / len(self.box_history))
        avg_w = int(sum(b[2] for b in self.box_history) / len(self.box_history))
        avg_h = int(sum(b[3] for b in self.box_history) / len(self.box_history))
        return [avg_x, avg_y, avg_w, avg_h]

    def get_body_frame(self, frame):
        h, w, _ = frame.shape

        # Convert to mediapipe Image format
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        # Use timestamp for VIDEO mode
        timestamp_ms = self.frame_index * 33  # ~30fps
        self.frame_index += 1

        result = self.pose_analyzer.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pts = result.pose_landmarks[0]
            self.latest_landmarks = pts

            visible_x, visible_y = self._filter_landmarks(pts)

            if len(visible_x) < 5:
                return None, None

            start_x = int(min(visible_x) * w)
            end_x   = int(max(visible_x) * w)
            start_y = int(min(visible_y) * h)
            end_y   = int(max(visible_y) * h)

            padding_x = int(0.05 * w)
            padding_y = int(0.05 * h)

            raw_box = [
                max(0, start_x - padding_x),
                max(0, start_y - padding_y),
                min(w, end_x - start_x + 2 * padding_x),
                min(h, end_y - start_y + 2 * padding_y)
            ]

            return self._smooth_box(raw_box), pts

        self.box_history.clear()
        return None, None