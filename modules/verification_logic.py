import math
from collections import deque


class PostureVerifier:
    """
    Per-person fall verifier with multi-frame confirmation and velocity tracking.
    """

    def __init__(self, confirmation_frames=5):
        self.confirmation_frames = confirmation_frames
        self.fall_counter   = 0
        self.normal_counter = 0
        self.hip_y_history  = deque(maxlen=10)

        self.RATIO_THRESHOLD  = 1.2
        self.ANGLE_MIN        = 45
        self.ANGLE_MAX        = 135
        self.VELOCITY_THRESHOLD = 0.02

    def _torso_angle(self, lms):
        s, h = lms[11], lms[23]
        return abs(math.degrees(math.atan2(h.y - s.y, h.x - s.x)))

    def _velocity(self, lms):
        self.hip_y_history.append(lms[23].y)
        if len(self.hip_y_history) < 3:
            return 0.0
        return self.hip_y_history[-1] - self.hip_y_history[-3]

    def evaluate_posture(self, box, landmarks):
        if box is None or landmarks is None:
            self.fall_counter = 0
            return False

        _, _, w, h = box
        if h == 0:
            return False

        ratio_ok  = (w / float(h)) > self.RATIO_THRESHOLD
        angle     = self._torso_angle(landmarks)
        inclined  = angle < self.ANGLE_MIN or angle > self.ANGLE_MAX
        dropping  = self._velocity(landmarks) > self.VELOCITY_THRESHOLD

        suspicious = (ratio_ok and inclined) or (dropping and inclined)

        if suspicious:
            self.fall_counter  += 1
            self.normal_counter = 0
        else:
            self.normal_counter += 1
            if self.normal_counter >= 3:
                self.fall_counter = 0

        return self.fall_counter >= self.confirmation_frames


# Single-person backward-compatible wrapper
_verifier = PostureVerifier()

def evaluate_posture(box, landmarks):
    return _verifier.evaluate_posture(box, landmarks)