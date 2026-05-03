import os
import cv2
from datetime import datetime
from collections import deque


class EventLogger:
    """
    Enhanced EventLogger with:
    - Pre-fall video buffer (saves clip BEFORE and AFTER the fall)
    - Screenshot evidence (same as before)
    - Clean structured log file
    - Console feedback with severity levels
    """

    def __init__(self, output_dir='output', buffer_seconds=5, fps=20):
        self.output_dir = output_dir
        self.fps = fps

        # Pre-fall buffer: stores last N frames before the fall
        buffer_size = buffer_seconds * fps
        self.frame_buffer = deque(maxlen=buffer_size)

        # Post-fall: how many frames to record after fall detected
        self.post_fall_frames = buffer_seconds * fps
        self.recording = False
        self.post_fall_counter = 0
        self.video_writer = None
        self.current_clip_path = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def buffer_frame(self, frame):
        """
        Call this every frame to keep a rolling buffer.
        This is what gives us the 'before the fall' footage.
        """
        self.frame_buffer.append(frame.copy())

        # If currently recording post-fall frames
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
            self.post_fall_counter -= 1

            if self.post_fall_counter <= 0:
                self._stop_recording()

    def _start_recording(self, frame):
        """Initialize video writer and write pre-fall buffer first."""
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_name = f"FALL_CLIP_{file_timestamp}.mp4"
        self.current_clip_path = os.path.join(self.output_dir, clip_name)

        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.current_clip_path, fourcc, self.fps, (w, h)
        )

        # Write pre-fall buffer frames first
        for buffered_frame in self.frame_buffer:
            self.video_writer.write(buffered_frame)

        self.recording = True
        self.post_fall_counter = self.post_fall_frames

    def _stop_recording(self):
        """Finalize and close the video file."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        self.post_fall_counter = 0
        print(f"[Logger] 🎬 Clip saved: {self.current_clip_path}")

    def log_event(self, message, frame=None, severity="INFO"):
        """
        Log an event with optional screenshot and video clip.

        severity options: INFO | WARNING | CRITICAL
        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M%S")

        # --- 1. Write to log file ---
        log_path = os.path.join(self.output_dir, 'system_logs.txt')
        with open(log_path, 'a') as f:
            f.write(f"[{timestamp}] [{severity}] {message}\n")

        # --- 2. Save screenshot if frame provided ---
        if frame is not None:
            screenshot_name = f"FALL_EVIDENCE_{file_timestamp}.jpg"
            screenshot_path = os.path.join(self.output_dir, screenshot_name)
            cv2.imwrite(screenshot_path, frame)

        # --- 3. Start video clip recording if CRITICAL and not already recording ---
        if severity == "CRITICAL" and frame is not None and not self.recording:
            self._start_recording(frame)

        # Console feedback
        icons = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}
        icon = icons.get(severity, "•")
        print(f"[Logger] {icon} [{severity}] {message}")


# ─── Backward-compatible wrapper ───────────────────────────────────────────────
# main.py calls log_event(message, frame) as a plain function.
# This keeps that working while using the new class internally.
_logger = EventLogger()

def log_event(message, frame=None):
    _logger.log_event(message, frame, severity="CRITICAL")

def buffer_frame(frame):
    """Call this every frame in main.py to enable pre-fall video buffer."""
    _logger.buffer_frame(frame)