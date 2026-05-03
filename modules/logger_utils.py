import os
import cv2
from datetime import datetime

def log_event(message, frame=None):
    """
    Logs system events to a text file and saves visual evidence (screenshots) 
    if a fall is detected. This ensures high system accountability.
    """
    # 1. Create the output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Generate a precise timestamp for the log entry
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 3. Append the event message to the log file
    log_path = os.path.join(output_dir, 'system_logs.txt')
    with open(log_path, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

    # 4. If a frame is provided (during a fall), save it as a screenshot
    if frame is not None:
        screenshot_name = f"FALL_EVIDENCE_{file_timestamp}.jpg"
        screenshot_path = os.path.join(output_dir, screenshot_name)
        cv2.imwrite(screenshot_path, frame)
        
    print(f"Log Updated: {message}") # Feedback for the console