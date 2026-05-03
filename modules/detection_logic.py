import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class PersonTracker:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pose_landmarker_lite.task')
        model_path = os.path.normpath(model_path)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_analyzer = vision.PoseLandmarker.create_from_options(options)

    def get_body_frame(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        analysis = self.pose_analyzer.detect(mp_image)

        if analysis.pose_landmarks and len(analysis.pose_landmarks) > 0:
            h, w, _ = frame.shape
            pts = analysis.pose_landmarks[0]

            all_x = [p.x for p in pts]
            all_y = [p.y for p in pts]

            start_x, end_x = int(min(all_x) * w), int(max(all_x) * w)
            start_y, end_y = int(min(all_y) * h), int(max(all_y) * h)

            box = [max(0, start_x), max(0, start_y), end_x - start_x, end_y - start_y]
            return box, pts

        return None, None