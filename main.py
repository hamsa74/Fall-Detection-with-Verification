import cv2
import os
# Import custom modules developed for the PUA AI roadmap
from modules.detection_logic import PersonTracker
from modules.verification_logic import evaluate_posture
from modules.logger_utils import log_event

def start_engine():
    """
    CareBot AI Main Engine: Supports Live Stream and Video Analysis.
    """
    print("\n--- CareBot AI System Settings ---")
    print("1. Live Camera Stream")
    print("2. Analyze Recorded Video File")
    
    choice = input("Select input source (1 or 2): ")

    if choice == '1':
        source = 0 
        print("[Status] Initializing system for Live Stream...")
    elif choice == '2':
        # Clean path from quotes and whitespace to avoid 'File Not Found'
        raw_path = input("Enter video file path: ").strip('"').strip("'").strip()
        video_path = os.path.normpath(raw_path)
        
        if not os.path.exists(video_path):
            print(f"[Error] File NOT found at: {video_path}")
            source = 0
        else:
            source = video_path
            print(f"[Status] Successfully loaded video: {video_path}")
    else:
        source = 0

    view = cv2.VideoCapture(source)
    if not view.isOpened():
        print("[Error] Failed to access the video source.")
        return

    # Initialize the Tracker with the Standard API
    tracker = PersonTracker()
    print("\n[System] CareBot AI is ACTIVE. Press 'q' to stop.\n")

    while True:
        success, img = view.read()
        if not success:
            break

        try:
            # Stage 1: Detection
            box, pts = tracker.get_body_frame(img)
            
            if box is not None:
                # Stage 2: Geometric Verification
                is_fall = evaluate_posture(box, pts)
                x, y, w, h = box
                
                if is_fall:
                    color = (0, 0, 255) # Red for Fall
                    cv2.putText(img, "CRITICAL: FALL DETECTED", (20, 50), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    log_event("Fall alert triggered", img)
                else:
                    color = (0, 255, 0) # Green for Normal
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        except Exception as e:
            continue

        cv2.imshow('CareBot AI - Monitor', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    view.release()
    cv2.destroyAllWindows()
    print("\n[System] Shutdown successful.")

if __name__ == "__main__":
    start_engine()