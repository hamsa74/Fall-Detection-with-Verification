import os
import cv2
from datetime import datetime
from collections import deque


class EventLogger:
    """
    Enhanced EventLogger with:
    - Pre/post fall video buffer
    - Screenshot evidence
    - Structured log file
    - Returns screenshot path for report integration
    """

    def __init__(self, output_dir='output', buffer_seconds=5, fps=20):
        self.output_dir = output_dir
        self.fps        = fps

        buffer_size       = buffer_seconds * fps
        self.frame_buffer = deque(maxlen=buffer_size)

        self.post_fall_frames   = buffer_seconds * fps
        self.recording          = False
        self.post_fall_counter  = 0
        self.video_writer       = None
        self.current_clip_path  = None

        os.makedirs(self.output_dir, exist_ok=True)

    def buffer_frame(self, frame):
        self.frame_buffer.append(frame.copy())

        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
            self.post_fall_counter -= 1
            if self.post_fall_counter <= 0:
                self._stop_recording()

    def _start_recording(self, frame):
        file_timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_name             = f"FALL_CLIP_{file_timestamp}.mp4"
        self.current_clip_path = os.path.join(self.output_dir, clip_name)

        h, w, _      = frame.shape
        fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.current_clip_path, fourcc, self.fps, (w, h)
        )
        for bf in self.frame_buffer:
            self.video_writer.write(bf)

        self.recording         = True
        self.post_fall_counter = self.post_fall_frames

    def _stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording        = False
        self.post_fall_counter = 0
        print(f"[Logger] Clip saved: {self.current_clip_path}")

    def log_event(self, message, frame=None, severity="INFO"):
        """Log event and return screenshot path (or None)."""
        now            = datetime.now()
        timestamp      = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M%S")
        screenshot_path = None

        # Log file
        log_path = os.path.join(self.output_dir, 'system_logs.txt')
        with open(log_path, 'a') as f:
            f.write(f"[{timestamp}] [{severity}] {message}\n")

        # Screenshot
        if frame is not None:
            screenshot_name = f"FALL_EVIDENCE_{file_timestamp}.jpg"
            screenshot_path = os.path.join(self.output_dir, screenshot_name)
            cv2.imwrite(screenshot_path, frame)

        # Start video clip
        if severity == "CRITICAL" and frame is not None and not self.recording:
            self._start_recording(frame)

        icons = {"INFO": "i", "WARNING": "!", "CRITICAL": "!!"}
        print(f"[Logger] [{icons.get(severity,'.')}] {message}")

        return screenshot_path


# Backward-compatible wrappers
_logger = EventLogger()

def log_event(message, frame=None):
    return _logger.log_event(message, frame, severity="CRITICAL")

def buffer_frame(frame):
    _logger.buffer_frame(frame)