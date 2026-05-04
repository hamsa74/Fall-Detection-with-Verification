import cv2
import numpy as np
from datetime import datetime
from collections import deque

class Dashboard:
    """
    Real-time monitoring dashboard rendered in a separate OpenCV window.
    Displays: FPS, Fall Count, Status, Confidence (weighted score), Time, Fall Timeline graph.
    """

    def __init__(self, width=400, height=600):
        self.width = width
        self.height = height
        self.window_name = 'CareBot AI - Dashboard'

        # Colors (BGR)
        self.BG      = (30, 30, 30)
        self.CARD_BG = (45, 45, 45)
        self.GREEN   = (80, 200, 120)
        self.RED     = (60, 60, 220)
        self.YELLOW  = (0, 200, 220)
        self.WHITE   = (220, 220, 220)
        self.MUTED   = (130, 130, 130)
        self.ACCENT  = (200, 140, 80)

        self.fps_history  = deque(maxlen=30)
        self.last_time    = datetime.now()
        self.fall_timeline = deque(maxlen=200)

        # Confidence smoothing — average last 10 values to avoid jitter
        self.conf_history = deque(maxlen=10)

        self.session_start  = datetime.now()
        self.last_fall_time = None

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)

    def _draw_card(self, canvas, x, y, w, h, label, value, value_color=None):
        if value_color is None:
            value_color = self.WHITE
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.CARD_BG, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (70,70,70), 1)
        cv2.putText(canvas, label, (x+12, y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.MUTED, 1)
        cv2.putText(canvas, str(value), (x+12, y+h-12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, value_color, 2)

    def _draw_confidence_bar(self, canvas, x, y, w, h, confidence):
        """
        Confidence bar using the new weighted score (0.0–1.0).
        Color: green < 0.4 | yellow < 0.7 | red >= 0.7
        Shows smoothed average to avoid flickering.
        """
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.CARD_BG, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (70,70,70), 1)
        cv2.putText(canvas, "Fall Confidence (weighted)", (x+12, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.MUTED, 1)

        bar_x = x + 12
        bar_y = y + 32
        bar_w = w - 80
        bar_h = 14

        # Smooth confidence
        self.conf_history.append(confidence)
        smooth_conf = sum(self.conf_history) / len(self.conf_history)

        # Background
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)

        # Fill
        fill_w = int(bar_w * smooth_conf)
        if fill_w > 0:
            if smooth_conf < 0.4:
                color = self.GREEN
            elif smooth_conf < 0.7:
                color = self.YELLOW
            else:
                color = self.RED
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), color, -1)

        # Percentage
        pct = f"{int(smooth_conf * 100)}%"
        cv2.putText(canvas, pct, (bar_x + bar_w + 8, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.WHITE, 1)

        # Breakdown labels
        cv2.putText(canvas, "Ratio 40%  Angle 40%  Velocity 20%",
                    (x+12, y+h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.33, self.MUTED, 1)

    def _draw_timeline(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.CARD_BG, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (70,70,70), 1)
        cv2.putText(canvas, "Fall timeline (last 200 frames)", (x+12, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.MUTED, 1)

        timeline = list(self.fall_timeline)
        if not timeline:
            return

        bar_area_x = x + 10
        bar_area_y = y + 28
        bar_area_w = w - 20
        bar_area_h = h - 38
        bar_w = max(1, bar_area_w // len(timeline))

        for i, val in enumerate(timeline):
            bx = bar_area_x + i * bar_w
            if val == 1:
                color = self.RED
                by    = bar_area_y
                bh    = bar_area_h
            else:
                color = (60, 80, 60)
                by    = bar_area_y + bar_area_h - 4
                bh    = 4
            cv2.rectangle(canvas, (bx, by), (bx+bar_w-1, by+bh), color, -1)

    def update(self, is_fall, fall_count, frame_count, confidence=0.0):
        now   = datetime.now()
        delta = (now - self.last_time).total_seconds()
        self.last_time = now
        fps = 1.0 / delta if delta > 0 else 0
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        self.fall_timeline.append(1 if is_fall else 0)
        if is_fall:
            self.last_fall_time = now

        canvas = np.full((self.height, self.width, 3), self.BG, dtype=np.uint8)

        # Header
        cv2.putText(canvas, "CareBot AI", (14, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, self.ACCENT, 2)
        cv2.putText(canvas, "Live Monitor", (14, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.MUTED, 1)
        cv2.line(canvas, (14, 60), (self.width-14, 60), (70,70,70), 1)

        # Status card
        status_text  = "!! FALL DETECTED !!" if is_fall else "Normal"
        status_color = self.RED if is_fall else self.GREEN
        border_color = (60,60,220) if is_fall else (40,100,60)
        cv2.rectangle(canvas, (14, 68), (self.width-14, 118), self.CARD_BG, -1)
        cv2.rectangle(canvas, (14, 68), (self.width-14, 118), border_color, 2)
        cv2.putText(canvas, "Status", (26, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.MUTED, 1)
        cv2.putText(canvas, status_text, (26, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, status_color, 2)

        # 2-col stat cards
        cw = (self.width - 36) // 2
        ch = 72

        self._draw_card(canvas, 14, 126, cw, ch,
                        "FPS", f"{avg_fps:.1f}", self.YELLOW)
        self._draw_card(canvas, 22+cw, 126, cw, ch,
                        "Falls detected", str(fall_count),
                        self.RED if fall_count > 0 else self.WHITE)

        self._draw_card(canvas, 14, 126+ch+8, cw, ch,
                        "Frames processed", str(frame_count), self.WHITE)

        elapsed    = int((now - self.session_start).total_seconds())
        mins, secs = divmod(elapsed, 60)
        self._draw_card(canvas, 22+cw, 126+ch+8, cw, ch,
                        "Session time", f"{mins:02d}:{secs:02d}", self.ACCENT)

        last_fall_str = self.last_fall_time.strftime("%H:%M:%S") if self.last_fall_time else "None"
        self._draw_card(canvas, 14, 126+(ch+8)*2, cw, ch,
                        "Last fall at", last_fall_str, self.YELLOW)

        fall_rate = (fall_count / frame_count * 100) if frame_count > 0 else 0
        self._draw_card(canvas, 22+cw, 126+(ch+8)*2, cw, ch,
                        "Fall rate", f"{fall_rate:.1f}%",
                        self.RED if fall_rate > 5 else self.GREEN)

        # Confidence bar (weighted score)
        conf_y = 126 + (ch+8)*3 + 4
        self._draw_confidence_bar(canvas, 14, conf_y, self.width-28, 68, confidence)

        # Timeline
        tl_y = conf_y + 76
        tl_h = self.height - tl_y - 14
        if tl_h > 50:
            self._draw_timeline(canvas, 14, tl_y, self.width-28, tl_h)

        cv2.imshow(self.window_name, canvas)

    def close(self):
        cv2.destroyWindow(self.window_name)