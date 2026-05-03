import math
from collections import deque

class PostureVerifier:
    """
    Advanced fall detection verifier - compatible with new mediapipe Tasks API.
    landmarks هنا list من NormalizedLandmark objects فيها x, y, z, visibility
    """

    def __init__(self, confirmation_frames=5):
        self.confirmation_frames = confirmation_frames
        self.fall_counter = 0
        self.normal_counter = 0
        self.hip_y_history = deque(maxlen=10)

        self.RATIO_THRESHOLD = 1.2
        self.ANGLE_MIN = 45
        self.ANGLE_MAX = 135
        self.VELOCITY_THRESHOLD = 0.02

    def _calculate_torso_angle(self, landmarks):
        shoulder = landmarks[11]
        hip = landmarks[23]
        dy = hip.y - shoulder.y
        dx = hip.x - shoulder.x
        return abs(math.degrees(math.atan2(dy, dx)))

    def _calculate_velocity(self, landmarks):
        hip_y = landmarks[23].y
        self.hip_y_history.append(hip_y)
        if len(self.hip_y_history) < 3:
            return 0.0
        return self.hip_y_history[-1] - self.hip_y_history[-3]

    def evaluate_posture(self, box_coords, landmarks):
        if box_coords is None or landmarks is None:
            self.fall_counter = 0
            return False

        _, _, width, height = box_coords
        if height == 0:
            return False

        posture_ratio = width / float(height)
        ratio_suspicious = posture_ratio > self.RATIO_THRESHOLD

        angle_deg = self._calculate_torso_angle(landmarks)
        is_inclined = angle_deg < self.ANGLE_MIN or angle_deg > self.ANGLE_MAX

        velocity = self._calculate_velocity(landmarks)
        is_dropping = velocity > self.VELOCITY_THRESHOLD

        is_suspicious = (ratio_suspicious and is_inclined) or (is_dropping and is_inclined)

        if is_suspicious:
            self.fall_counter += 1
            self.normal_counter = 0
        else:
            self.normal_counter += 1
            if self.normal_counter >= 3:
                self.fall_counter = 0

        return self.fall_counter >= self.confirmation_frames


# Backward-compatible wrapper
_verifier = PostureVerifier()

def evaluate_posture(box_coords, landmarks):
    return _verifier.evaluate_posture(box_coords, landmarks)