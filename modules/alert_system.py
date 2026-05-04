import threading
import sys


def _beep():
    """
    Cross-platform system beep.
    Windows  → winsound
    Linux/Mac → print bell character
    """
    try:
        if sys.platform == 'win32':
            import winsound
            # 3 beeps: frequency=1000Hz, duration=300ms each
            for _ in range(3):
                winsound.Beep(1000, 300)
        else:
            # Terminal bell
            for _ in range(3):
                sys.stdout.write('\a')
                sys.stdout.flush()
    except Exception as e:
        print(f"[Alert] Beep failed: {e}")


def play_fall_alert():
    """
    Play fall alert sound in a background thread.
    Non-blocking — won't slow down the main video loop.
    """
    t = threading.Thread(target=_beep, daemon=True)
    t.start()