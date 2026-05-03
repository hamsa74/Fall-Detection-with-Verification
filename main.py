import cv2
import os
from modules.detection_logic import PersonTracker
from modules.verification_logic import evaluate_posture
from modules.logger_utils import log_event

def start_engine():
    """
    The main execution core of the CareBot AI system.
    Integrates Pose Estimation, Geometric Analysis, and Real-time Logging.
    """
    view = cv2.VideoCapture(0)
    if not view.isOpened():
        print("Error: Camera hardware not accessible.")
        return

    tracker = PersonTracker()
    print("--- CareBot AI: Monitoring System Active ---")
    print("Press 'q' to stop and save logs.")

    while True:
        success, img = view.read()
        if not success:
            break

        try:
            box, pts = tracker.get_body_frame(img)
            
            if box is not None:
                is_fall = evaluate_posture(box, pts)
                x, y, w, h = box
                
                if is_fall:
                    color = (0, 0, 255) 
                    thickness = 3
                    cv2.putText(img, "CRITICAL: FALL DETECTED", (20, 50), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    
                    log_event("Fall alert triggered!", img)
                else:
                    color = (0, 255, 0)
                    thickness = 2
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        except Exception as e:
            continue

        cv2.imshow('CareBot AI - Fall Detection System', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    view.release()
    cv2.destroyAllWindows()
    print("System shut down safely. All logs are stored in the /output folder.")

if __name__ == "__main__":
    start_engine()