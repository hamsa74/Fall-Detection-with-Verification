import cv2
import os
import json
import math
from datetime import datetime
from modules.detection_logic import PersonTracker
from modules.verification_logic import PostureVerifier


def run_evaluation(video_path: str, output_dir: str = 'output'):
    """
    Interactive evaluation tool:
    - Shows video frame by frame
    - User presses SPACE to mark ground truth fall frames
    - System runs detection in parallel
    - Computes Precision, Recall, F1, Accuracy
    - Saves results to JSON + readable TXT report
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Evaluator] Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 20

    print("\n╔══════════════════════════════════════════╗")
    print("║     CareBot AI — Evaluation Tool         ║")
    print("╠══════════════════════════════════════════╣")
    print("║  SPACE  → Mark current frame as FALL     ║")
    print("║  Q      → Quit and compute results       ║")
    print("║  P      → Pause / Resume                 ║")
    print("╚══════════════════════════════════════════╝\n")

    tracker   = PersonTracker(max_persons=3)
    verifiers = {}

    # Ground truth and predictions per frame
    gt_labels   = {}   # {frame_num: 1}  — user-marked falls
    pred_labels = {}   # {frame_num: 1}  — system predictions

    frame_count = 0
    paused      = False
    prev_fall_states = {}

    cv2.namedWindow('CareBot Evaluator', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CareBot Evaluator', 900, 540)

    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1

            # System detection
            any_fall = False
            try:
                if frame_count % 2 == 0:
                    persons = tracker.get_persons(frame)
                    for p in persons:
                        pid = p['id']
                        if pid not in verifiers:
                            verifiers[pid] = PostureVerifier(confirmation_frames=5)
                        is_fall, conf, tif, _ = verifiers[pid].evaluate_posture(p['box'], p['landmarks'])
                        if is_fall:
                            any_fall = True
                    prev_fall_states = any_fall
                else:
                    any_fall = prev_fall_states if isinstance(prev_fall_states, bool) else False
            except:
                pass

            if any_fall:
                pred_labels[frame_count] = 1

        # Draw UI on frame
        display = frame.copy()
        h, w    = display.shape[:2]

        # Progress bar
        progress = frame_count / total_frames if total_frames > 0 else 0
        cv2.rectangle(display, (0, h-8), (w, h), (30,30,30), -1)
        cv2.rectangle(display, (0, h-8), (int(w*progress), h), (249,115,22), -1)

        # Info overlay
        cv2.rectangle(display, (0,0), (w, 52), (0,0,0), -1)
        cv2.putText(display, f"Frame: {frame_count}/{total_frames}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        cv2.putText(display, f"GT Falls marked: {len(gt_labels)}  |  System detected: {len(pred_labels)}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        # System detection indicator
        sys_color = (0,0,220) if any_fall else (0,200,80)
        sys_text  = "SYSTEM: FALL" if any_fall else "SYSTEM: Normal"
        cv2.putText(display, sys_text, (w-200, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, sys_color, 2)

        # GT marker indicator
        if frame_count in gt_labels:
            cv2.putText(display, "GT: FALL ✓", (w-200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,220), 2)

        if paused:
            cv2.putText(display, "PAUSED — Press P to resume", (10, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,220), 2)

        cv2.imshow('CareBot Evaluator', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' '):
            # Mark current frame as ground truth fall
            gt_labels[frame_count] = 1
            print(f"[GT] Frame #{frame_count} marked as FALL")

    cap.release()
    cv2.destroyAllWindows()

    # ── Compute metrics ──────────────────────────────────────
    all_frames = set(range(1, frame_count + 1))

    TP = len(set(pred_labels) & set(gt_labels))
    FP = len(set(pred_labels) - set(gt_labels))
    FN = len(set(gt_labels)   - set(pred_labels))
    TN = len(all_frames - set(pred_labels) - set(gt_labels))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (TP + TN) / frame_count if frame_count > 0 else 0

    results = {
        "video":       os.path.basename(video_path),
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_frames":frame_count,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1_score":    round(f1,        4),
        "accuracy":    round(accuracy,  4),
        "gt_falls":    len(gt_labels),
        "pred_falls":  len(pred_labels),
    }

    # Save JSON
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"eval_{ts}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save readable TXT report
    txt_path = os.path.join(output_dir, f"eval_{ts}.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("   CareBot AI — Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Video:          {results['video']}\n")
        f.write(f"Date:           {results['timestamp']}\n")
        f.write(f"Total Frames:   {results['total_frames']}\n\n")
        f.write("-" * 50 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  True Positives  (TP): {TP}\n")
        f.write(f"  False Positives (FP): {FP}\n")
        f.write(f"  False Negatives (FN): {FN}\n")
        f.write(f"  True Negatives  (TN): {TN}\n\n")
        f.write("-" * 50 + "\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Precision : {precision*100:.2f}%\n")
        f.write(f"  Recall    : {recall*100:.2f}%\n")
        f.write(f"  F1 Score  : {f1*100:.2f}%\n")
        f.write(f"  Accuracy  : {accuracy*100:.2f}%\n\n")
        f.write("=" * 50 + "\n")

    # Print summary
    print("\n" + "="*50)
    print("   CareBot AI — Evaluation Results")
    print("="*50)
    print(f"  TP: {TP}  FP: {FP}  FN: {FN}  TN: {TN}")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {txt_path}")
    print("="*50 + "\n")

    return results


if __name__ == "__main__":
    path = input("Enter video path: ").strip('"').strip("'").strip()
    run_evaluation(path)