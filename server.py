import cv2
import base64
import asyncio
import threading
from datetime import datetime
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from modules.detection_logic import PersonTracker
from modules.verification_logic import PostureVerifier
from modules.logger_utils import log_event, buffer_frame
from modules.alert_system import play_fall_alert

app = FastAPI(title="CareBot AI Server")

# ── Global state ────────────────────────────────────────────
state = {
    "running":     False,
    "source":      None,
    "frame_count": 0,
    "fall_count":  0,
    "fps":         0.0,
    "persons":     [],        # list of {id, is_fall, confidence}
    "any_fall":    False,
}

connected_clients: list[WebSocket] = []
_engine_thread: threading.Thread | None = None


# ── WebSocket broadcast ─────────────────────────────────────
async def broadcast(data: dict):
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(data)
        except:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


# ── AI Engine (runs in background thread) ───────────────────
def run_engine(source, loop: asyncio.AbstractEventLoop):
    tracker   = PersonTracker(max_persons=3)
    verifiers = {}

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "error", "msg": "Cannot open video source"}), loop
        )
        return

    fps_history      = deque(maxlen=30)
    last_time        = datetime.now()
    prev_fall_states = {}
    frame_count      = 0
    fall_count       = 0

    state["running"] = True

    while state["running"]:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        buffer_frame(frame)

        # FPS
        now   = datetime.now()
        delta = (now - last_time).total_seconds()
        last_time = now
        if delta > 0:
            fps_history.append(1.0 / delta)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        fall_states = {}
        confidences = {}
        persons_out = []
        any_fall    = False

        try:
            if frame_count % 2 == 0:
                persons = tracker.get_persons(frame)
                for p in persons:
                    pid = p['id']
                    if pid not in verifiers:
                        verifiers[pid] = PostureVerifier(confirmation_frames=5)
                    verifier            = verifiers[pid]
                    is_fall, confidence = verifier.evaluate_posture(p['box'], p['landmarks'])
                    fall_states[pid]    = is_fall
                    confidences[pid]    = confidence

                    was_fall = prev_fall_states.get(pid, False)
                    if is_fall and not was_fall:
                        fall_count += 1
                        play_fall_alert()
                        log_event(f"Person #{pid+1} Fall alert #{fall_count}",
                                  frame, person_id=pid+1,
                                  frame_number=frame_count, confidence=confidence)

                    persons_out.append({
                        "id":         pid + 1,
                        "is_fall":    is_fall,
                        "confidence": round(confidence * 100),
                    })

                prev_fall_states = {p['id']: fall_states.get(p['id'], False)
                                    for p in persons}
                any_fall = any(fall_states.values())

            # Encode frame to base64 JPEG for streaming
            _, buf    = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            b64_frame = base64.b64encode(buf).decode('utf-8')

            payload = {
                "type":        "frame",
                "frame":       b64_frame,
                "frame_count": frame_count,
                "fall_count":  fall_count,
                "fps":         round(avg_fps, 1),
                "any_fall":    any_fall,
                "persons":     persons_out,
            }

            asyncio.run_coroutine_threadsafe(broadcast(payload), loop)

        except Exception as e:
            print(f"[Engine] Error: {e}")

    cap.release()
    state["running"] = False
    asyncio.run_coroutine_threadsafe(
        broadcast({"type": "stopped", "fall_count": fall_count}), loop
    )
    print("[Engine] Stopped.")


# ── REST endpoints ───────────────────────────────────────────
@app.post("/start/camera")
async def start_camera():
    if state["running"]:
        return {"status": "already running"}
    loop = asyncio.get_event_loop()
    t = threading.Thread(target=run_engine, args=(0, loop), daemon=True)
    t.start()
    return {"status": "started", "source": "camera"}


@app.post("/stop")
async def stop_engine():
    state["running"] = False
    return {"status": "stopping"}


@app.get("/status")
async def get_status():
    return {
        "running":     state["running"],
        "frame_count": state["frame_count"],
        "fall_count":  state["fall_count"],
        "fps":         state["fps"],
    }


# ── WebSocket ────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[WS] Client connected. Total: {len(connected_clients)}")
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "start_camera":
                if not state["running"]:
                    loop = asyncio.get_event_loop()
                    t = threading.Thread(
                        target=run_engine, args=(0, loop), daemon=True
                    )
                    t.start()

            elif data.get("action") == "stop":
                state["running"] = False

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(connected_clients)}")


# ── Serve PWA static files ───────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n[Server] CareBot AI Server starting...")
    print(f"[Server] Open on PC:     http://localhost:8000")
    print(f"[Server] Open on Mobile: http://{local_ip}:8000")
    print(f"[Server] Make sure PC and Mobile are on the same WiFi!\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)