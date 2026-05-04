import math
from collections import deque


class PostureVerifier:
    """
    Per-person fall verifier with multi-frame confirmation,
    velocity tracking, and a detailed confidence score (0.0 - 1.0).

    Confidence is a weighted blend of 3 signals:
      - ratio_score   : how horizontal the bounding box is
      - angle_score   : how inclined the torso is
      - velocity_score: how fast the person is dropping
    """

    def __init__(self, confirmation_frames=5):
        self.confirmation_frames = confirmation_frames
        self.fall_counter        = 0
        self.normal_counter      = 0
        self.hip_y_history       = deque(maxlen=10)

        self.RATIO_THRESHOLD     = 1.2
        self.ANGLE_MIN           = 45
        self.ANGLE_MAX           = 135
        self.VELOCITY_THRESHOLD  = 0.02

        # Last computed confidence (public)
        self.confidence          = 0.0

    def _torso_angle(self, lms):
        s, h = lms[11], lms[23]
        return abs(math.degrees(math.atan2(h.y - s.y, h.x - s.x)))

    def _velocity(self, lms):
        self.hip_y_history.append(lms[23].y)
        if len(self.hip_y_history) < 3:
            return 0.0
        return self.hip_y_history[-1] - self.hip_y_history[-3]

    def _compute_confidence(self, box, landmarks):
        """
        Returns a float 0.0-1.0 representing fall confidence.
        Weighted blend: ratio 40% + angle 40% + velocity 20%
        """
        _, _, w, h = box
        if h == 0:
            return 0.0

        # --- Ratio score (0-1) ---
        ratio = w / float(h)
        ratio_score = min(max((ratio - 0.8) / (2.0 - 0.8), 0.0), 1.0)

        # --- Angle score (0-1) ---
        angle = self._torso_angle(landmarks)
        # Perfect upright = 90 deg → score 0. Horizontal = 0 or 180 → score 1
        deviation = min(abs(angle - 90), 90)   # 0..90
        angle_score = deviation / 90.0

        # --- Velocity score (0-1) ---
        velocity = self._velocity(landmarks)
        velocity_score = min(max(velocity / (self.VELOCITY_THRESHOLD * 3), 0.0), 1.0)

        # Weighted blend
        confidence = (ratio_score * 0.4) + (angle_score * 0.4) + (velocity_score * 0.2)
        return round(min(confidence, 1.0), 3)

    def evaluate_posture(self, box, landmarks):
        """
        Returns (is_fall: bool, confidence: float 0-1).
        """
        if box is None or landmarks is None:
            self.fall_counter = 0
            self.confidence   = 0.0
            return False, 0.0

        _, _, w, h = box
        if h == 0:
            return False, 0.0

        # Raw signals
        ratio_ok = (w / float(h)) > self.RATIO_THRESHOLD
        angle    = self._torso_angle(landmarks)
        inclined = angle < self.ANGLE_MIN or angle > self.ANGLE_MAX
        dropping = self._velocity(landmarks) > self.VELOCITY_THRESHOLD

        suspicious = (ratio_ok and inclined) or (dropping and inclined)

        if suspicious:
            self.fall_counter  += 1
            self.normal_counter = 0
        else:
            self.normal_counter += 1
            if self.normal_counter >= 3:
                self.fall_counter = 0

        is_fall          = self.fall_counter >= self.confirmation_frames
        self.confidence  = self._compute_confidence(box, landmarks)
        return is_fall, self.confidence


# Single-person backward-compatible wrapper
_verifier = PostureVerifier()

def evaluate_posture(box, landmarks):
    is_fall, confidence = _verifier.evaluate_posture(box, landmarks)
    return is_fall