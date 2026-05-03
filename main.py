import cv2
import os
from datetime import datetime
from collections import deque
from modules.detection_logic import PersonTracker
from modules.verification_logic import PostureVerifier
from modules.logger_utils import log_event, buffer_frame
from modules.dashboard import Dashboard
from modules.report_generator import generate_report


def get_video_source():
    print("\n--- CareBot AI System Settings ---")
    print("1. Live Camera Stream")
    print("2. Analyze Recorded Video File")

    choice = input("Select input source (1 or 2): ").strip()

    if choice == '1':
        print("[Status] Initializing system for Live Stream...")
        return 0
    elif choice == '2':
        raw_path = input("Enter video file path: ").strip('"').strip("'").strip()
        video_path = os.path.normpath(raw_path)
        if not os.path.exists(video_path):
            print(f"[Error] File NOT found at: {video_path}")
            return None
        print(f"[Status] Successfully loaded video: {video_path}")
        return video_path
    else:
        print("[Error] Invalid choice. Please select 1 or 2.")
        return None


def draw_ui(img, persons, fall_states, frame_count, fall_count):
    for p in persons:
        pid   = p['id']
        box   = p['box']
        color = p['color']
        x, y, w, h = box

        is_fall    = fall_states.get(pid, False)
        draw_color = (0, 0, 255) if is_fall else color

        cv2.rectangle(img, (x, y), (x+w, y+h), draw_color, 2)

        label = f"Person #{pid+1} - FALL!" if is_fall else f"Person #{pid+1}"
        cv2.rectangle(img, (x, y-24), (x + len(label)*10, y), (0,0,0), -1)
        cv2.putText(img, label, (x+4, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)

    stats = f"Frames: {frame_count}  |  People: {len(persons)}  |  Falls: {fall_count}"
    cv2.putText(img, stats, (20, img.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    return img


def start_engine():
    source = get_video_source()
    if source is None:
        return

    view = cv2.VideoCapture(source)
    if not view.isOpened():
        print("[Error] Failed to access the video source.")
        return

    tracker   = PersonTracker(max_persons=3)
    dashboard = Dashboard(width=400, height=640)
    verifiers = {}

    # Session tracking
    session_start  = datetime.now()
    frame_count    = 0
    fall_count     = 0
    fall_events    = []       # list of fall dicts for report
    fall_timeline  = []       # 0/1 per frame
    fps_history    = deque(maxlen=60)
    last_time      = datetime.now()
    prev_fall_states = {}

    print("\n[System] CareBot AI is ACTIVE. Press 'q' to stop.\n")

    while True:
        success, img = view.read()
        if not success:
            print("[System] End of video stream.")
            break

        frame_count += 1
        buffer_frame(img)

        # FPS tracking
        now   = datetime.now()
        delta = (now - last_time).total_seconds()
        last_time = now
        if delta > 0:
            fps_history.append(1.0 / delta)

        fall_states    = {}
        max_confidence = 0.0
        any_fall       = False

        try:
            persons = tracker.get_persons(img)

            for p in persons:
                pid = p['id']

                if pid not in verifiers:
                    verifiers[pid] = PostureVerifier(confirmation_frames=5)

                verifier   = verifiers[pid]
                is_fall    = verifier.evaluate_posture(p['box'], p['landmarks'])
                confidence = min(verifier.fall_counter / verifier.confirmation_frames, 1.0)

                fall_states[pid] = is_fall
                max_confidence   = max(max_confidence, confidence)

                was_fall = prev_fall_states.get(pid, False)
                if is_fall and not was_fall:
                    fall_count += 1
                    ts = datetime.now().strftime("%H:%M:%S")

                    # Save screenshot
                    screenshot_path = log_event(
                        f"Person #{pid+1} Fall alert #{fall_count}", img
                    )

                    # Store for report
                    fall_events.append({
                        'id':        fall_count,
                        'person_id': pid,
                        'timestamp': ts,
                        'frame':     frame_count,
                        'screenshot': screenshot_path or '',
                    })

            prev_fall_states = {p['id']: fall_states.get(p['id'], False) for p in persons}
            any_fall = any(fall_states.values())
            img = draw_ui(img, persons, fall_states, frame_count, fall_count)

        except Exception as e:
            print(f"[Warning] Frame error: {e}")

        fall_timeline.append(1 if any_fall else 0)

        dashboard.update(
            is_fall    = any_fall,
            fall_count = fall_count,
            frame_count= frame_count,
            confidence = max_confidence
        )

        cv2.imshow('CareBot AI - Monitor', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Session ended ---
    session_end = datetime.now()
    avg_fps     = sum(fps_history) / len(fps_history) if fps_history else 0

    view.release()
    dashboard.close()
    cv2.destroyAllWindows()

    print(f"\n[System] Shutdown successful.")
    print(f"[System] Total frames: {frame_count} | Total falls: {fall_count}")

    # Generate HTML report
    print("[Report] Generating session report...")
    generate_report({
        'start_time':   session_start,
        'end_time':     session_end,
        'total_frames': frame_count,
        'fps':          avg_fps,
        'falls':        fall_events,
        'fall_timeline':fall_timeline,
        'video_source': source,
    })


if __name__ == "__main__":
    start_engine()