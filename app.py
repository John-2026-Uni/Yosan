from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import sqlite3
import hashlib
import binascii
import secrets
import uvicorn
import os
import cv2
import time as time_module
from datetime import datetime
import threading

# ---------- CONFIG ----------
DB_FILE = "users.db"
ITERATIONS = 100_000
SALT_BYTES = 16
HASH_BYTES = 32

# Motion detection tuning
MOTION_THRESHOLD      = 25      # pixel diff threshold (lower = more sensitive)
MIN_CONTOUR_RATIO     = 0.005   # minimum contour as fraction of frame area (0.5%)
MOTION_CONFIRM_FRAMES = 3       # consecutive frames needed to confirm motion
BG_LEARN_RATE         = 0.05    # how fast background adapts (0=static, 1=instant)
MOTION_COOLDOWN_SECS  = 5       # seconds to wait before logging a new event

app = FastAPI()

# Development CORS (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Permissions-Policy"] = "camera=*, microphone=*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ---------- Motion detection globals ----------
static_back = None
motion_list = [None, None]
camera_active = False
motion_detected = False
current_frame = None
camera_thread = None
frame_lock = threading.Lock()
motion_lock = threading.Lock()

# Camera source: 0 = local webcam, or an RTSP/HTTP URL string
camera_source = 0

# ---------- DB helpers ----------
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS login_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        name TEXT,
        status TEXT NOT NULL,
        ip TEXT,
        logged_at TEXT DEFAULT (datetime('now','localtime'))
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS motion_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT NOT NULL,
        end_time TEXT,
        duration_seconds REAL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS camera_config(
        id INTEGER PRIMARY KEY CHECK (id = 1),
        source TEXT NOT NULL DEFAULT '0',
        label TEXT,
        updated_at TEXT DEFAULT (datetime('now','localtime'))
    )
    """)
    # Seed default row (local webcam index 0)
    cursor.execute(
        "INSERT OR IGNORE INTO camera_config (id, source, label) VALUES (1, '0', 'Local Webcam')"
    )
    conn.commit()
    conn.close()

init_db()

# Load persisted camera source from DB
def _load_camera_source():
    global camera_source
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT source FROM camera_config WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            src = row["source"]
            camera_source = int(src) if src.isdigit() else src
    except Exception as e:
        print(f"[startup] Could not load camera source: {e}")

_load_camera_source()

# ---------- crypto helpers ----------
def generate_salt() -> str:
    return binascii.hexlify(secrets.token_bytes(SALT_BYTES)).decode()

def hash_password(password: str, salt_hex: str) -> str:
    salt = binascii.unhexlify(salt_hex.encode())
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, ITERATIONS, dklen=HASH_BYTES)
    return binascii.hexlify(dk).decode()

# ---------- Motion detection helpers ----------
def log_motion_event(start: datetime, end: datetime):
    try:
        duration = (end - start).total_seconds()
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO motion_events (start_time, end_time, duration_seconds) VALUES (?, ?, ?)",
            (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), round(duration, 2))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[motion_log error] {e}")

def _make_error_frame(msg: str):
    """Return a dark frame with an error message — shown while reconnecting."""
    import numpy as np
    frame = np.zeros((360, 640, 3), dtype="uint8")
    cv2.putText(frame, "⚠ Camera Unavailable", (60, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
    cv2.putText(frame, msg, (60, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (60, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return frame


def _open_capture(src):
    """Open a cv2.VideoCapture with settings suited to the source type."""
    is_rtsp = isinstance(src, str) and src.lower().startswith(("rtsp://", "rtsps://"))
    is_http = isinstance(src, str) and src.lower().startswith(("http://", "https://"))

    if is_rtsp:
        # Force TCP transport — more reliable than default UDP on most networks
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(src)

    if is_rtsp or is_http:
        # Give network streams time to negotiate
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000)

    return cap


# Shared error state so /api/camera/start can report why it failed
camera_error: str = ""


def motion_detection_loop():
    global static_back, motion_list, camera_active, current_frame, motion_detected, camera_error

    # Resolve source
    src = camera_source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    MAX_RECONNECT   = 10          # give up after this many consecutive failures
    RECONNECT_DELAY = 5           # seconds between reconnect attempts
    reconnect_count = 0
    camera_error    = ""

    print(f"[camera] Opening source: {src!r}")

    while camera_active:
        video = _open_capture(src)

        if not video.isOpened():
            reconnect_count += 1
            msg = f"Could not open source '{src}' (attempt {reconnect_count}/{MAX_RECONNECT})"
            print(f"[camera] {msg}")
            camera_error = msg

            if reconnect_count >= MAX_RECONNECT:
                print("[camera] Max reconnect attempts reached. Giving up.")
                camera_active = False
                return

            # Write an error frame to keep the UI informed
            err_frame = _make_error_frame(f"Reconnecting… ({reconnect_count}/{MAX_RECONNECT})")
            with frame_lock:
                current_frame = err_frame

            for _ in range(int(RECONNECT_DELAY / 0.2)):
                if not camera_active:
                    return
                time_module.sleep(0.2)
            continue

        print(f"[camera] Stream opened successfully: {src!r}")
        camera_error    = ""
        reconnect_count = 0

        static_back  = None
        motion_list  = [None, None]
        local_times  = []
        motion_counter  = 0
        last_motion_end = None
        consecutive_failures = 0

        while camera_active:
            check, frame = video.read()
            if not check:
                consecutive_failures += 1
                if consecutive_failures >= 30:   # ~1.5 s of bad reads
                    print("[camera] Too many failed reads — reconnecting.")
                    break
                time_module.sleep(0.05)
                continue

            consecutive_failures = 0

        motion = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Improvement: resolution-adaptive blur kernel (always odd, scales with width)
        h_px, w_px = gray.shape
        blur_k = max(5, (w_px // 30) | 1)
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        # Improvement: initialise background as float for weighted accumulation
        if static_back is None:
            static_back = gray.astype("float")
            with frame_lock:
                current_frame = frame.copy()
            continue

        # Improvement: adaptive background — slowly drifts with lighting changes
        cv2.accumulateWeighted(gray, static_back, BG_LEARN_RATE)
        bg_snap = cv2.convertScaleAbs(static_back)

        diff_frame  = cv2.absdiff(bg_snap, gray)
        # Improvement: configurable threshold instead of hardcoded 30
        thresh_frame = cv2.threshold(diff_frame, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cnts, _ = cv2.findContours(
            thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Improvement: resolution-relative minimum contour area
        frame_area = h_px * w_px
        min_area   = frame_area * MIN_CONTOUR_RATIO
        contour_detected = False

        for contour in cnts:
            if cv2.contourArea(contour) < min_area:
                continue
            contour_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Improvement: require N consecutive frames before confirming motion
        if contour_detected:
            motion_counter = min(motion_counter + 1, MOTION_CONFIRM_FRAMES)
        else:
            motion_counter = max(motion_counter - 1, 0)

        motion = 1 if motion_counter >= MOTION_CONFIRM_FRAMES else 0

        # Overlay status text on frame
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if motion:
            cv2.putText(frame, "⚠ MOTION DETECTED", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "● Monitoring", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 80), 2)
        cv2.putText(frame, now_str, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Track motion start / end
        motion_list.append(motion)
        motion_list = motion_list[-2:]

        if motion_list[-1] == 1 and motion_list[-2] == 0:
            # Improvement: cooldown — skip logging if too soon after last event
            now = datetime.now()
            if last_motion_end is None or (now - last_motion_end).total_seconds() >= MOTION_COOLDOWN_SECS:
                local_times.append(now)
            else:
                print(f"[motion] Cooldown active, skipping event start.")

        if motion_list[-1] == 0 and motion_list[-2] == 1:
            end_t = datetime.now()
            local_times.append(end_t)
            last_motion_end = end_t
            if len(local_times) >= 2:
                log_motion_event(local_times[-2], local_times[-1])

        with motion_lock:
            motion_detected = bool(motion)

        with frame_lock:
            current_frame = frame.copy()

        # ── end inner while (per-frame) ──

        # If motion was active when the inner loop exited
        with motion_lock:
            if motion_detected and len(local_times) % 2 == 1:
                log_motion_event(local_times[-1], datetime.now())
            motion_detected = False

        static_back = None
        video.release()

        if camera_active:
            print("[camera] Stream dropped — attempting reconnect in 5 s.")
            err_frame = _make_error_frame("Stream lost — reconnecting…")
            with frame_lock:
                current_frame = err_frame
            reconnect_count += 1
            if reconnect_count >= MAX_RECONNECT:
                print("[camera] Max reconnects reached.")
                camera_active = False
                return
            for _ in range(int(RECONNECT_DELAY / 0.2)):
                if not camera_active:
                    break
                time_module.sleep(0.2)

    # ── end outer while (reconnect) ──
    print("[camera] Stopped.")

def generate_mjpeg():
    """Yield MJPEG frames from the global current_frame buffer."""
    while camera_active:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None

        if frame is None:
            time_module.sleep(0.05)
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )
        time_module.sleep(0.033)   # ~30 fps cap

# ---------- request models ----------
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    repass: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ---------- Auth endpoints ----------
@app.post("/api/signup")
def signup(data: SignupRequest):
    if data.password != data.repass:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (data.email.lower(),))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered.")
    salt = generate_salt()
    pw_hash = hash_password(data.password, salt)
    cursor.execute(
        "INSERT INTO users (name, email, password_hash, salt) VALUES (?, ?, ?, ?)",
        (data.name.strip(), data.email.lower(), pw_hash, salt)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Account created."}

@app.post("/api/login")
def login(data: LoginRequest, request: Request):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name, password_hash, salt FROM users WHERE email = ?", (data.email.lower(),))
    row = cursor.fetchone()
    ip = request.client.host if request.client else "unknown"
    if not row:
        cursor.execute(
            "INSERT INTO login_logs (email, name, status, ip) VALUES (?, NULL, 'FAILED', ?)",
            (data.email.lower(), ip)
        )
        conn.commit()
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    name, stored_hash, salt = row["name"], row["password_hash"], row["salt"]
    if hash_password(data.password, salt) == stored_hash:
        cursor.execute(
            "INSERT INTO login_logs (email, name, status, ip) VALUES (?, ?, 'SUCCESS', ?)",
            (data.email.lower(), name, ip)
        )
        conn.commit()
        conn.close()
        return {"status": "ok", "name": name}
    cursor.execute(
        "INSERT INTO login_logs (email, name, status, ip) VALUES (?, ?, 'FAILED', ?)",
        (data.email.lower(), name, ip)
    )
    conn.commit()
    conn.close()
    raise HTTPException(status_code=401, detail="Invalid email or password.")

# ---------- Camera endpoints ----------
@app.post("/api/camera/start")
def camera_start():
    global camera_active, camera_thread, camera_error

    if camera_active:
        return {"status": "already_running"}

    # Validate source before spawning thread
    src = camera_source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    is_network = isinstance(src, str) and src.lower().startswith(
        ("rtsp://", "rtsps://", "http://", "https://")
    )

    # For local device index: do a quick open/close check
    if not is_network:
        test = cv2.VideoCapture(src)
        opened = test.isOpened()
        test.release()
        if not opened:
            hint = (
                "No camera found at device index. "
                "If using a network CCTV, go to Camera Settings and enter the RTSP/HTTP URL."
            )
            raise HTTPException(status_code=400, detail=hint)

    # For network streams: don't block the HTTP request with a full connection attempt;
    # the thread will handle reconnection and surface errors via camera_error.
    camera_error  = ""
    camera_active = True
    camera_thread = threading.Thread(target=motion_detection_loop, daemon=True)
    camera_thread.start()
    return {"status": "ok", "source_type": "network" if is_network else "local", "source": str(camera_source)}

@app.post("/api/camera/stop")
def camera_stop():
    global camera_active
    camera_active = False
    return {"status": "ok"}

@app.get("/api/camera/status")
def camera_status():
    with motion_lock:
        md = motion_detected
    return {"active": camera_active, "motion": md, "error": camera_error}

@app.get("/api/camera/stream")
def camera_stream():
    if not camera_active:
        raise HTTPException(status_code=503, detail="Camera not running.")
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/motion/events")
def motion_events(limit: int = 100):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, start_time, end_time, duration_seconds FROM motion_events ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(1 for r in rows if r["start_time"].startswith(today))
    return {"events": rows, "today_count": today_count, "total": len(rows)}


# ---------- Camera config model ----------
class CameraConfigRequest(BaseModel):
    source: str          # "0" for local webcam, or RTSP/HTTP URL
    label: str = ""      # friendly name, e.g. "Front Door"


# ---------- Camera config endpoints ----------
@app.get("/api/camera/config")
def get_camera_config():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT source, label, updated_at FROM camera_config WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"source": row["source"], "label": row["label"], "updated_at": row["updated_at"]}
    return {"source": "0", "label": "Local Webcam", "updated_at": None}


@app.post("/api/camera/config")
def set_camera_config(data: CameraConfigRequest):
    global camera_source
    if camera_active:
        raise HTTPException(status_code=409, detail="Stop the camera before changing its source.")

    # Basic validation: must be a digit (local index) or a URL
    src = data.source.strip()
    if not src:
        raise HTTPException(status_code=400, detail="Source cannot be empty.")
    if not src.isdigit() and not src.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Source must be a device index (e.g. '0') or a URL starting with rtsp://, http://, or https://"
        )

    camera_source = src
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE camera_config SET source=?, label=?, updated_at=datetime('now','localtime') WHERE id=1",
        (src, data.label.strip())
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "source": src, "label": data.label.strip()}



# ---------- Admin endpoint ----------
@app.get("/api/admin/logs")
def admin_logs(limit: int = 100):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, email, name, status, ip, logged_at FROM login_logs ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"logs": rows}

# ---------- Shared style ----------
YOSAN_BASE_STYLE = """
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    background: linear-gradient(145deg, #EEF4FF 0%, #DBEAFE 50%, #BAE6FD 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
  }

  .card {
    background: rgba(255, 255, 255, 0.90);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(147, 197, 253, 0.4);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    width: 100%;
    max-width: 440px;
    box-shadow: 0 8px 40px rgba(37, 99, 235, 0.10), 0 1px 3px rgba(37, 99, 235, 0.06);
  }

  .brand {
    text-align: center;
    margin-bottom: 1.75rem;
  }

  .brand h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #1D4ED8;
    line-height: 1;
    letter-spacing: -2px;
  }

  .brand p {
    font-size: 0.8rem;
    color: #fff;
    background: linear-gradient(90deg, #2563EB, #38BDF8);
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    margin-top: 8px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
  }

  label {
    display: block;
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  input {
    display: block;
    width: 100%;
    padding: 11px 16px;
    border: 1.5px solid #BFDBFE;
    border-radius: 10px;
    font-size: 0.95rem;
    color: #1E293B;
    background: #F8FAFF;
    outline: none;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    font-family: 'DM Sans', sans-serif;
  }

  input:focus {
    border-color: #2563EB;
    background: #fff;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
  }

  .btn-primary {
    display: block;
    width: 100%;
    padding: 13px;
    border-radius: 10px;
    font-size: 0.95rem;
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    border: none;
    background: linear-gradient(135deg, #2563EB 0%, #38BDF8 100%);
    color: #fff;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    transition: all 0.15s;
    margin-top: 6px;
    text-decoration: none;
    text-align: center;
    text-transform: uppercase;
  }

  .btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
  }

  .btn-primary:active {
    transform: translateY(1px);
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
  }

  .btn-outline {
    display: block;
    width: 100%;
    padding: 12px;
    border-radius: 10px;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    border: 1.5px solid #BFDBFE;
    background: transparent;
    color: #2563EB;
    text-align: center;
    text-decoration: none;
    transition: all 0.15s;
    margin-top: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .btn-outline:hover {
    background: #EFF6FF;
    border-color: #2563EB;
  }

  .msg {
    margin-top: 14px;
    font-size: 0.88rem;
    padding: 10px 14px;
    border-radius: 10px;
    display: none;
    font-weight: 500;
  }

  .msg.success { background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; display: block; }
  .msg.error   { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; display: block; }

  .footer {
    margin-top: 1.25rem;
    font-size: 0.88rem;
    color: #94A3B8;
    text-align: center;
    font-weight: 500;
  }

  .footer a { color: #2563EB; text-decoration: none; font-weight: 600; }
  .footer a:hover { text-decoration: underline; }

  .divider {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.2rem 0;
    color: #CBD5E1;
    font-size: 0.82rem;
  }

  .divider::before,
  .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E2E8F0;
  }
"""

# ---------- Frontend HTML ----------

LANDING_HTML = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Yosan</title>
  <style>
    {YOSAN_BASE_STYLE}
    .card {{ text-align: center; }}
    .brand h1 {{ font-size: 4rem; }}
    .tagline {{
      font-size: 0.95rem; color: #64748B; margin-bottom: 2rem;
      font-weight: 400; line-height: 1.6;
    }}
    .hero-icon {{
      width: 64px; height: 64px;
      background: linear-gradient(135deg, #DBEAFE, #BAE6FD);
      border-radius: 18px; display: flex; align-items: center;
      justify-content: center; margin: 0 auto 1.2rem;
      font-size: 1.8rem; box-shadow: 0 4px 14px rgba(37,99,235,0.15);
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="hero-icon">🎥</div>
    <div class="brand">
      <h1>Yosan</h1>
      <p>CCTV MONITORING</p>
    </div>
    <p class="tagline">Intelligent CCTV monitoring with motion detection.<br>Create an account or sign in to begin.</p>
    <a class="btn-primary" href="/signup">Create Account</a>
    <a class="btn-outline" href="/login">Sign In</a>
  </div>
</body>
</html>
"""

SIGNUP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Sign Up — Yosan</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; font-family: 'DM Sans', sans-serif; }
    body {
      background: linear-gradient(145deg, #EEF4FF 0%, #DBEAFE 50%, #BAE6FD 100%);
      display: flex; flex-direction: column;
      align-items: center; justify-content: center; min-height: 100vh;
    }
    .topbar {
      position: fixed; top: 0; left: 0; right: 0; height: 60px;
      background: rgba(255,255,255,0.92); backdrop-filter: blur(12px);
      border-bottom: 1px solid rgba(147,197,253,0.4);
      box-shadow: 0 1px 12px rgba(37,99,235,0.08);
      display: flex; align-items: center; padding: 0 2rem; z-index: 10;
    }
    .topbar-logo { display: flex; align-items: center; gap: 10px; }
    .topbar-logo-icon {
      width: 32px; height: 32px;
      background: linear-gradient(135deg, #2563EB, #38BDF8);
      border-radius: 8px; display: flex; align-items: center;
      justify-content: center; font-size: 1rem;
    }
    .topbar-logo span {
      font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.3rem;
      font-weight: 800; color: #1D4ED8; letter-spacing: -1px;
    }
    .page {
      display: flex; width: min(1000px, 96vw); min-height: 580px;
      border-radius: 18px; overflow: hidden;
      box-shadow: 0 20px 60px rgba(37,99,235,0.15), 0 1px 3px rgba(37,99,235,0.06);
      margin-top: 76px; margin-bottom: 24px;
    }
    .left {
      width: 48%;
      background: linear-gradient(160deg, #1D4ED8 0%, #2563EB 60%, #0EA5E9 100%);
      border-radius: 18px 0 0 18px; padding: 2.8rem 2.6rem 2.2rem;
      display: flex; flex-direction: column; justify-content: center;
      position: relative; overflow: hidden;
    }
    .left::before {
      content: ''; position: absolute; top: -80px; right: -80px;
      width: 260px; height: 260px; background: rgba(255,255,255,0.07); border-radius: 50%;
    }
    .left::after {
      content: ''; position: absolute; bottom: -60px; left: -40px;
      width: 180px; height: 180px; background: rgba(255,255,255,0.05); border-radius: 50%;
    }
    .panel-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 1.6rem; }
    .panel-brand-icon {
      width: 44px; height: 44px; background: rgba(255,255,255,0.18);
      border-radius: 12px; display: flex; align-items: center;
      justify-content: center; font-size: 1.4rem; border: 1px solid rgba(255,255,255,0.25);
    }
    .panel-brand-text .name {
      font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.5rem;
      font-weight: 800; color: #fff; letter-spacing: -1px;
    }
    .panel-brand-text .sub {
      font-size: 0.72rem; color: rgba(255,255,255,0.6); margin-top: 1px;
      text-transform: uppercase; letter-spacing: 0.6px;
    }
    .panel-header { margin-bottom: 1.6rem; }
    .panel-header h2 {
      font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.9rem;
      font-weight: 800; color: #fff; letter-spacing: -1.5px; line-height: 1.1;
    }
    .panel-header p { font-size: 0.88rem; color: rgba(255,255,255,0.6); margin-top: 4px; }
    .field-group { margin-bottom: 0.8rem; }
    .field-group input {
      width: 100%; padding: 12px 18px;
      background: rgba(255,255,255,0.12);
      border: 1.5px solid rgba(255,255,255,0.25);
      border-radius: 10px; font-family: 'DM Sans', sans-serif;
      font-size: 0.95rem; color: #fff; outline: none;
      transition: background 0.15s, border-color 0.15s;
    }
    .field-group input::placeholder { color: rgba(255,255,255,0.45); }
    .field-group input:focus {
      background: rgba(255,255,255,0.2); border-color: rgba(255,255,255,0.6);
      box-shadow: 0 0 0 3px rgba(255,255,255,0.1);
    }
    .checks { margin: 0.5rem 0 1rem; }
    .check-row {
      display: flex; align-items: flex-start; gap: 8px; margin-bottom: 6px;
      font-family: 'DM Sans', sans-serif; font-size: 0.82rem;
      color: rgba(255,255,255,0.75); cursor: pointer;
    }
    .check-row input[type=checkbox] {
      display: inline-block; width: 16px !important; height: 16px !important;
      margin-bottom: 0 !important; padding: 0 !important; border: none !important;
      border-radius: 4px !important; background: none !important;
      box-shadow: none !important; accent-color: #38BDF8;
      flex-shrink: 0; margin-top: 2px; cursor: pointer;
    }
    .check-row a { color: #BAE6FD; text-decoration: underline; }
    .submit-btn {
      width: 100%; display: block; padding: 13px; background: #fff;
      border: none; border-radius: 10px; font-family: 'DM Sans', sans-serif;
      font-size: 0.95rem; font-weight: 700; color: #1D4ED8; letter-spacing: 0.3px;
      cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,0.15);
      transition: all 0.15s; text-transform: uppercase;
    }
    .submit-btn:hover  { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,0,0,0.2); }
    .submit-btn:active { transform: translateY(1px); box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .msg {
      margin-top: 10px; font-size: 0.85rem; padding: 9px 14px;
      border-radius: 8px; display: none; font-weight: 500; text-align: center;
    }
    .msg.success { background: rgba(16,185,129,0.2); color: #6EE7B7; border: 1px solid rgba(110,231,183,0.3); display: block; }
    .msg.error   { background: rgba(239,68,68,0.2); color: #FCA5A5; border: 1px solid rgba(252,165,165,0.3); display: block; }
    .panel-footer { margin-top: 1rem; font-size: 0.82rem; color: rgba(255,255,255,0.5); text-align: center; }
    .panel-footer a { color: #BAE6FD; font-weight: 600; text-decoration: none; }
    .panel-footer a:hover { text-decoration: underline; }
    .right {
      flex: 1; background: #fff; border-radius: 0 18px 18px 0; position: relative;
      overflow: hidden; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 1.5rem; padding: 2.5rem;
    }
    .right-title {
      font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2rem;
      font-weight: 800; color: #1D4ED8; letter-spacing: -1.5px;
      text-align: center; z-index: 1;
    }
    .right-sub { font-size: 0.9rem; color: #64748B; text-align: center; line-height: 1.6; max-width: 280px; z-index: 1; }
    .features { display: flex; flex-direction: column; gap: 0.8rem; z-index: 1; width: 100%; max-width: 260px; }
    .feature-item {
      display: flex; align-items: center; gap: 12px; padding: 12px 16px;
      background: #F0F7FF; border-radius: 12px; border: 1px solid #DBEAFE;
    }
    .feature-icon {
      width: 36px; height: 36px;
      background: linear-gradient(135deg, #2563EB, #38BDF8);
      border-radius: 8px; display: flex; align-items: center;
      justify-content: center; font-size: 1rem; flex-shrink: 0;
    }
    .feature-text { font-size: 0.85rem; font-weight: 600; color: #1E293B; }
    .blob { position: absolute; opacity: 0.06; border-radius: 50%; background: #2563EB; }
    .blob-1 { width: 320px; height: 320px; top: -100px; right: -100px; }
    .blob-2 { width: 200px; height: 200px; bottom: -60px; left: -40px; }
    .link-btn {
      background: none; border: none; padding: 0;
      font-family: 'DM Sans', sans-serif; font-size: inherit;
      color: #BAE6FD; text-decoration: underline; cursor: pointer;
    }
    .link-btn:hover { color: #fff; }
    .modal-backdrop {
      display: none; position: fixed; inset: 0;
      background: rgba(15,23,42,0.6); backdrop-filter: blur(4px);
      z-index: 9999; align-items: center; justify-content: center;
    }
    .modal-backdrop.open { display: flex; }
    .modal-box {
      background: #fff; border-radius: 16px; width: min(520px, 92vw);
      max-height: 80vh; display: flex; flex-direction: column;
      box-shadow: 0 20px 60px rgba(15,23,42,0.25); border: 1px solid rgba(147,197,253,0.4);
      animation: modalIn 0.2s ease;
    }
    @keyframes modalIn { from { opacity:0; transform:scale(0.96) translateY(12px); } to { opacity:1; transform:scale(1) translateY(0); } }
    .modal-header { display:flex; align-items:center; justify-content:space-between; padding:1.2rem 1.5rem 0.9rem; border-bottom:1px solid #E2E8F0; }
    .modal-header h2 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.2rem; font-weight:700; color:#1E293B; letter-spacing:-0.5px; }
    .modal-close { background:#F1F5F9; border:none; font-size:1.2rem; color:#64748B; cursor:pointer; line-height:1; padding:4px 8px; border-radius:6px; transition:all 0.1s; }
    .modal-close:hover { background:#FEE2E2; color:#DC2626; }
    .modal-body { padding:1.2rem 1.5rem; overflow-y:auto; flex:1; }
    .modal-body h3 { font-family:'DM Sans',sans-serif; font-size:0.85rem; font-weight:700; color:#2563EB; text-transform:uppercase; letter-spacing:0.5px; margin:1rem 0 0.3rem; }
    .modal-body h3:first-child { margin-top:0; }
    .modal-body p { font-family:'DM Sans',sans-serif; font-size:0.88rem; color:#475569; line-height:1.6; }
    .modal-footer { padding:0.9rem 1.5rem 1.2rem; border-top:1px solid #E2E8F0; display:flex; justify-content:flex-end; }
    .modal-done { padding:10px 28px; background:linear-gradient(135deg,#2563EB,#38BDF8); border:none; border-radius:8px; font-family:'DM Sans',sans-serif; font-size:0.9rem; font-weight:700; color:#fff; cursor:pointer; box-shadow:0 4px 12px rgba(37,99,235,0.3); transition:all 0.15s; text-transform:uppercase; letter-spacing:0.3px; }
    .modal-done:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(37,99,235,0.4); }
    @media (max-width: 700px) { .page { flex-direction: column; width: 96vw; } .left { width: 100%; border-radius: 18px 18px 0 0; } .right { display: none; } }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-logo">
      <div class="topbar-logo-icon">🎥</div>
      <span>Yosan</span>
    </div>
  </div>
  <div class="page">
    <div class="left">
      <div class="panel-brand">
        <div class="panel-brand-icon">🎥</div>
        <div class="panel-brand-text">
          <div class="name">Yosan</div>
          <div class="sub">CCTV MONITORING</div>
        </div>
      </div>
      <div class="panel-header">
        <h2>Create Account</h2>
        <p>Start monitoring in seconds.</p>
      </div>
      <div class="field-group"><input id="username" type="text" placeholder="Full Name" autocomplete="username"/></div>
      <div class="field-group"><input id="email" type="email" placeholder="Email Address" autocomplete="email"/></div>
      <div class="field-group"><input id="password" type="password" placeholder="Password" autocomplete="new-password"/></div>
      <div class="field-group"><input id="repass" type="password" placeholder="Confirm Password" autocomplete="new-password"/></div>
      <div class="checks">
        <div class="check-row">
          <input type="checkbox" id="terms"/>
          <label for="terms">I agree to the&nbsp;<button type="button" class="link-btn" onclick="openModal('termsModal')">Terms and Conditions</button></label>
        </div>
        <div class="check-row">
          <input type="checkbox" id="privacy"/>
          <label for="privacy">I have read the&nbsp;<button type="button" class="link-btn" onclick="openModal('privacyModal')">Privacy Policy</button></label>
        </div>
      </div>
      <button class="submit-btn" onclick="doSignup()">Create Account</button>
      <div id="msg" class="msg"></div>
      <div class="panel-footer">Already have an account? <a href="/login">Sign in</a> &nbsp;·&nbsp; <a href="/">Home</a></div>
    </div>
    <div class="right">
      <div class="blob blob-1"></div>
      <div class="blob blob-2"></div>
      <div class="right-title">Welcome to<br>Yosan</div>
      <p class="right-sub">Monitor your CCTV cameras with intelligent motion detection.</p>
      <div class="features">
        <div class="feature-item"><div class="feature-icon">🌐</div><span class="feature-text">Live CCTV Monitoring</span></div>
        <div class="feature-item"><div class="feature-icon">🎯</div><span class="feature-text">Motion Detection</span></div>
        <div class="feature-item"><div class="feature-icon">🔒</div><span class="feature-text">Secure Access Logs</span></div>
      </div>
    </div>
  </div>

  <div id="termsModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="termsTitle">
    <div class="modal-box">
      <div class="modal-header"><h2 id="termsTitle">Terms and Conditions</h2><button class="modal-close" onclick="closeModal('termsModal')">&times;</button></div>
      <div class="modal-body">
        <h3>1. Acceptance of Terms</h3><p>By creating an account and using Yosan, you agree to be bound by these Terms and Conditions.</p>
        <h3>2. Use of Service</h3><p>Yosan is a CCTV monitoring tool. You agree to use the service only for lawful purposes.</p>
        <h3>3. Account Responsibility</h3><p>You are responsible for maintaining the confidentiality of your account credentials.</p>
        <h3>4. Camera & Privacy</h3><p>You are solely responsible for ensuring camera usage complies with local privacy laws. Do not monitor spaces without consent.</p>
        <h3>5. Modifications</h3><p>We reserve the right to modify these terms at any time.</p>
        <h3>6. Termination</h3><p>We may suspend or terminate your account at our discretion if you violate these terms.</p>
      </div>
      <div class="modal-footer"><button class="modal-done" onclick="acceptAndClose('termsModal','terms')">I Agree</button></div>
    </div>
  </div>

  <div id="privacyModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="privacyTitle">
    <div class="modal-box">
      <div class="modal-header"><h2 id="privacyTitle">Privacy Policy</h2><button class="modal-close" onclick="closeModal('privacyModal')">&times;</button></div>
      <div class="modal-body">
        <h3>1. Information We Collect</h3><p>We collect your name, email, and system activity logs. We do not sell your data to third parties.</p>
        <h3>2. Camera Data</h3><p>Camera streams and motion detection data are processed locally on your server and are not transmitted to Anthropic or any third party.</p>
        <h3>3. Data Security</h3><p>We take reasonable measures to protect your data, including password hashing and secure storage.</p>
        <h3>4. Your Rights</h3><p>You may request deletion of your account and associated data at any time.</p>
        <h3>5. Changes</h3><p>We may update this Privacy Policy periodically.</p>
      </div>
      <div class="modal-footer"><button class="modal-done" onclick="acceptAndClose('privacyModal','privacy')">I Have Read This</button></div>
    </div>
  </div>

<script>
function openModal(id) { document.getElementById(id).classList.add('open'); document.body.style.overflow='hidden'; }
function closeModal(id) { document.getElementById(id).classList.remove('open'); document.body.style.overflow=''; }
function acceptAndClose(modalId, checkboxId) { document.getElementById(checkboxId).checked=true; closeModal(modalId); }
document.addEventListener('click', function(e) { if(e.target.classList.contains('modal-backdrop')){ e.target.classList.remove('open'); document.body.style.overflow=''; } });
document.addEventListener('keydown', function(e) { if(e.key==='Escape'){ document.querySelectorAll('.modal-backdrop.open').forEach(m=>{ m.classList.remove('open'); document.body.style.overflow=''; }); } });

async function doSignup() {
  const msg = document.getElementById('msg');
  msg.className = 'msg';
  if (!document.getElementById('terms').checked || !document.getElementById('privacy').checked) {
    msg.className = 'msg error'; msg.textContent = 'Please accept the terms and privacy policy.'; return;
  }
  const payload = { name: document.getElementById('username').value, email: document.getElementById('email').value, password: document.getElementById('password').value, repass: document.getElementById('repass').value };
  try {
    const res = await fetch('/api/signup', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const data = await res.json();
    if (!res.ok) throw data;
    msg.className = 'msg success'; msg.textContent = 'Account created! Redirecting…';
    setTimeout(() => { window.location.href = '/login'; }, 900);
  } catch(e) { msg.className = 'msg error'; msg.textContent = e.detail || 'Something went wrong.'; }
}
</script>
</body>
</html>
"""

LOGIN_HTML = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Sign In — Yosan</title>
  <style>
    {YOSAN_BASE_STYLE}
    .card {{ position: relative; overflow: hidden; }}
    .card::before {{
      content: ''; position: absolute; top: -40px; right: -40px;
      width: 180px; height: 180px;
      background: radial-gradient(circle, rgba(37,99,235,0.06), transparent 70%);
      border-radius: 50%; pointer-events: none;
    }}
    .hero-icon {{
      width: 52px; height: 52px;
      background: linear-gradient(135deg, #DBEAFE, #BAE6FD);
      border-radius: 14px; display: flex; align-items: center;
      justify-content: center; font-size: 1.4rem;
      box-shadow: 0 4px 12px rgba(37,99,235,0.12); margin-bottom: 1rem;
    }}
    .brand h1 {{ font-size: 2.2rem; letter-spacing: -1.5px; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="hero-icon">🔐</div>
    <div class="brand">
      <h1>Welcome back</h1>
      <p>Sign in to Yosan</p>
    </div>
    <label for="loginEmail">Email Address</label>
    <input id="loginEmail" type="email" placeholder="you@email.com" />
    <label for="loginPass">Password</label>
    <input id="loginPass" type="password" placeholder="Your password" />
    <button class="btn-primary" onclick="doLogin()">Sign In</button>
    <div id="out" class="msg"></div>
    <div class="footer">No account yet? <a href="/signup">Create one</a> &nbsp;·&nbsp; <a href="/">Home</a></div>
  </div>
<script>
async function doLogin() {{
  const out = document.getElementById('out');
  out.className = 'msg';
  const payload = {{ email: document.getElementById('loginEmail').value, password: document.getElementById('loginPass').value }};
  try {{
    const res = await fetch('/api/login', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload) }});
    const data = await res.json();
    if (!res.ok) throw data;
    localStorage.setItem('yosan_user', data.name);
    out.className = 'msg success';
    out.textContent = 'Welcome back, ' + data.name + '! Redirecting…';
    setTimeout(() => {{ window.location.href = '/dashboard'; }}, 700);
  }} catch(e) {{
    out.className = 'msg error';
    out.textContent = e.detail || 'Something went wrong.';
  }}
}}
</script>
</body>
</html>
"""

DASHBOARD_HTML = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Dashboard — Yosan</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      height: 100%; font-family: 'DM Sans', sans-serif;
      background: linear-gradient(145deg, #EEF4FF 0%, #DBEAFE 50%, #BAE6FD 100%);
      min-height: 100vh;
    }}
    .shell {{ display: flex; height: 100vh; overflow: hidden; }}

    /* ── Sidebar ── */
    .sidebar {{
      width: 240px; flex-shrink: 0;
      background: linear-gradient(180deg, #1D4ED8 0%, #2563EB 60%, #1E40AF 100%);
      box-shadow: 4px 0 20px rgba(29,78,216,0.25);
      display: flex; flex-direction: column; position: relative; z-index: 2;
    }}
    .sidebar-brand {{
      display: flex; align-items: center; gap: 10px;
      padding: 1.5rem 1.4rem 1.2rem;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .sidebar-brand-icon {{
      width: 36px; height: 36px; background: rgba(255,255,255,0.15);
      border-radius: 10px; display: flex; align-items: center;
      justify-content: center; font-size: 1.1rem; border: 1px solid rgba(255,255,255,0.2);
    }}
    .sidebar-brand-text .name {{
      font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.2rem;
      font-weight: 800; color: #fff; letter-spacing: -0.8px;
    }}
    .sidebar-brand-text .sub {{
      font-size: 0.68rem; color: rgba(255,255,255,0.5); margin-top: 1px;
      text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .nav-section {{ padding: 1rem 0.8rem 0.4rem; }}
    .nav-section-label {{
      font-size: 0.65rem; font-weight: 700; color: rgba(255,255,255,0.35);
      text-transform: uppercase; letter-spacing: 1px; padding: 0 0.6rem;
    }}
    .nav-item {{
      display: flex; align-items: center; gap: 10px;
      padding: 0.75rem 1rem; cursor: pointer;
      background: transparent; border-radius: 10px; margin: 2px 0.4rem;
      text-decoration: none; transition: background 0.15s;
      border: none; width: calc(100% - 0.8rem); color: rgba(255,255,255,0.65);
    }}
    .nav-item:hover {{ background: rgba(255,255,255,0.1); color: #fff; }}
    .nav-item.active {{
      background: rgba(255,255,255,0.15); color: #fff;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.15);
    }}
    .nav-item .nav-label {{ font-family: 'DM Sans', sans-serif; font-size: 0.88rem; font-weight: 600; }}
    .nav-icon {{ font-size: 1rem; width: 22px; text-align: center; flex-shrink: 0; }}
    .sidebar-spacer {{ flex: 1; }}
    .sidebar-logout {{ padding: 1rem 1.2rem 1.2rem; border-top: 1px solid rgba(255,255,255,0.1); }}
    .sidebar-logout a {{
      display: flex; align-items: center; gap: 8px; padding: 10px 14px;
      border-radius: 10px; background: rgba(255,255,255,0.08);
      color: rgba(255,255,255,0.7); font-family: 'DM Sans', sans-serif;
      font-size: 0.88rem; font-weight: 600; text-decoration: none; transition: all 0.15s;
    }}
    .sidebar-logout a:hover {{ background: rgba(255,255,255,0.15); color: #fff; }}

    /* ── Main ── */
    .main {{ flex: 1; display: flex; flex-direction: column; overflow-y: auto; }}
    .topbar {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 1rem 2rem; background: rgba(255,255,255,0.9);
      backdrop-filter: blur(12px); border-bottom: 1px solid rgba(147,197,253,0.3);
      box-shadow: 0 1px 8px rgba(37,99,235,0.06); position: sticky; top: 0; z-index: 1;
    }}
    .topbar-title {{ font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.15rem; font-weight: 700; color: #1E293B; letter-spacing: -0.5px; }}
    .topbar-user {{ display: flex; align-items: center; gap: 10px; }}
    .topbar-user-avatar {{
      width: 34px; height: 34px;
      background: linear-gradient(135deg, #2563EB, #38BDF8);
      border-radius: 50%; display: flex; align-items: center;
      justify-content: center; font-size: 0.85rem; font-weight: 700; color: #fff;
    }}
    .topbar-user-name {{ font-family: 'DM Sans', sans-serif; font-size: 0.9rem; font-weight: 600; color: #334155; }}

    /* ── Views ── */
    .view {{ display: none; padding: 1.8rem 2rem 2rem; flex: 1; }}
    .view.active {{ display: block; }}

    /* ── Panels ── */
    .panel {{
      background: rgba(255,255,255,0.85); border: 1px solid rgba(147,197,253,0.3);
      border-radius: 16px; box-shadow: 0 4px 16px rgba(37,99,235,0.07); padding: 1.5rem;
    }}
    .panel-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.2rem; }}
    .panel-title {{ font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1rem; font-weight: 700; color: #1E293B; letter-spacing: -0.3px; }}

    .stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px,1fr)); gap: 0.9rem; margin-bottom: 1.4rem; }}
    .stat-card {{
      background: #fff; border: 1px solid rgba(147,197,253,0.35);
      border-radius: 14px; box-shadow: 0 2px 8px rgba(37,99,235,0.06); padding: 1.1rem 1.2rem;
    }}
    .stat-label {{ font-family: 'DM Sans', sans-serif; font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }}
    .stat-value {{ font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.9rem; font-weight: 800; color: #1E293B; letter-spacing: -1.5px; line-height: 1.1; margin-top: 4px; }}
    .stat-icon {{ font-size: 1.3rem; margin-bottom: 4px; }}

    .refresh-btn {{
      padding: 7px 18px; background: linear-gradient(135deg, #2563EB, #38BDF8);
      border: none; border-radius: 8px; font-family: 'DM Sans', sans-serif;
      font-size: 0.82rem; font-weight: 700; color: #fff; cursor: pointer;
      box-shadow: 0 2px 8px rgba(37,99,235,0.3); transition: all 0.15s;
    }}
    .refresh-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37,99,235,0.4); }}

    .filter-row {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }}
    .filter-btn {{
      padding: 5px 16px; border-radius: 8px; border: 1.5px solid #BFDBFE;
      background: #fff; font-family: 'DM Sans', sans-serif; font-size: 0.8rem;
      font-weight: 600; color: #2563EB; cursor: pointer; transition: all 0.12s;
    }}
    .filter-btn.active, .filter-btn:hover {{ background: #2563EB; border-color: #2563EB; color: #fff; }}

    .log-table {{ width: 100%; border-collapse: collapse; }}
    .log-table th {{
      text-align: left; font-family: 'DM Sans', sans-serif; font-size: 0.75rem;
      font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.4px;
      padding-bottom: 10px; border-bottom: 1.5px solid #E2E8F0;
    }}
    .log-table td {{
      padding: 10px 0; font-size: 0.87rem; color: #334155; font-weight: 500;
      border-bottom: 1px solid #F1F5F9;
    }}
    .log-table tr:last-child td {{ border-bottom: none; }}
    .log-table tr:hover td {{ background: #F8FAFF; }}
    .badge {{
      display: inline-block; padding: 3px 10px; border-radius: 6px;
      font-size: 0.72rem; font-weight: 700; font-family: 'DM Sans', sans-serif;
      text-transform: uppercase; letter-spacing: 0.3px;
    }}
    .badge-success {{ background: #ECFDF5; color: #059669; }}
    .badge-failed  {{ background: #FEF2F2; color: #DC2626; }}
    .badge-motion  {{ background: #FFF7ED; color: #EA580C; }}
    .no-logs {{ text-align: center; padding: 2rem; font-family: 'DM Sans', sans-serif; color: #94A3B8; font-size: 0.9rem; }}

    /* ── Camera view ── */
    .cam-wrap {{ display: flex; flex-direction: column; align-items: center; gap: 1.2rem; }}
    .cam-video-box {{
      width: 100%; max-width: 700px; border-radius: 14px;
      border: 2px solid #BFDBFE; background: #0F172A; overflow: hidden;
      box-shadow: 0 6px 24px rgba(37,99,235,0.12);
      min-height: 200px; display: flex; align-items: center; justify-content: center;
      position: relative;
    }}
    #cam-stream {{ width: 100%; display: block; border-radius: 12px; }}
    .cam-placeholder {{ font-family: 'DM Sans', sans-serif; color: #38BDF8; font-size: 0.95rem; padding: 3rem; text-align: center; opacity: 0.7; }}
    .motion-indicator {{
      position: absolute; top: 12px; right: 12px;
      background: rgba(0,0,0,0.6); border-radius: 8px;
      padding: 5px 12px; font-family: 'DM Sans', sans-serif;
      font-size: 0.8rem; font-weight: 700; color: #fff;
      display: none;
    }}
    .motion-indicator.visible {{ display: block; }}
    .motion-indicator.alert {{ background: rgba(220,38,38,0.85); animation: pulse 1s infinite; }}
    .motion-indicator.ok {{ background: rgba(5,150,105,0.85); }}
    @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.6; }} }}
    .cam-controls {{ display: flex; gap: 0.8rem; flex-wrap: wrap; justify-content: center; }}
    .cam-btn {{
      padding: 10px 22px; border-radius: 8px; border: none;
      background: linear-gradient(135deg, #2563EB, #38BDF8); color: #fff;
      font-family: 'DM Sans', sans-serif; font-size: 0.9rem; font-weight: 600;
      cursor: pointer; box-shadow: 0 3px 10px rgba(37,99,235,0.3); transition: all 0.15s;
    }}
    .cam-btn:hover {{ transform: translateY(-1px); box-shadow: 0 5px 14px rgba(37,99,235,0.4); }}
    .cam-btn:active {{ transform: translateY(1px); }}
    .cam-btn-red {{ background: linear-gradient(135deg, #EF4444, #F87171); box-shadow: 0 3px 10px rgba(239,68,68,0.3); }}
    .cam-btn-red:hover {{ box-shadow: 0 5px 14px rgba(239,68,68,0.4); }}
    .cam-btn-green {{ background: linear-gradient(135deg, #059669, #34D399); box-shadow: 0 3px 10px rgba(5,150,105,0.3); }}
    #cam-status {{ font-family: 'DM Sans', sans-serif; font-size: 0.88rem; color: #64748B; min-height: 1.4rem; font-weight: 500; }}

    @media (max-width: 700px) {{
      .sidebar {{ width: 56px; }}
      .nav-label, .sidebar-brand-text, .nav-section-label {{ display: none; }}
      .sidebar-logout a {{ font-size: 0; padding: 10px 0; justify-content: center; }}
    }}
  </style>
</head>
<body>
<div class="shell">

  <aside class="sidebar" aria-label="Sidebar navigation">
    <div class="sidebar-brand">
      <div class="sidebar-brand-icon">🎥</div>
      <div class="sidebar-brand-text">
        <div class="name">Yosan</div>
        <div class="sub">admin panel</div>
      </div>
    </div>
    <div class="nav-section">
      <div class="nav-section-label">Main Menu</div>
    </div>
    <button class="nav-item active" id="nav-camera" onclick="showView('camera')">
      <span class="nav-icon">📷</span>
      <span class="nav-label">Camera</span>
    </button>
    <button class="nav-item" id="nav-motion" onclick="showView('motion')">
      <span class="nav-icon">🎯</span>
      <span class="nav-label">Motion Events</span>
    </button>
    <button class="nav-item" id="nav-logs" onclick="showView('logs')">
      <span class="nav-icon">🛡️</span>
      <span class="nav-label">Login Logs</span>
    </button>
    <button class="nav-item" id="nav-settings" onclick="showView('settings')">
      <span class="nav-icon">⚙️</span>
      <span class="nav-label">Camera Settings</span>
    </button>
    <div class="sidebar-spacer"></div>
    <div class="sidebar-logout">
      <a href="/" onclick="stopCamera(); localStorage.removeItem('yosan_user')">
        <span>🚪</span> Sign Out
      </a>
    </div>
  </aside>

  <div class="main">
    <header class="topbar">
      <span class="topbar-title" id="page-title">Camera</span>
      <div class="topbar-user">
        <div class="topbar-user-avatar" id="user-avatar">?</div>
        <span class="topbar-user-name" id="username">friend</span>
      </div>
    </header>

    <!-- ── View: Camera ── -->
    <div class="view active" id="view-camera">
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-icon">🔴</div>
          <div class="stat-label">Camera Status</div>
          <div class="stat-value" id="stat-cam-status" style="font-size:1.1rem;margin-top:8px;">Offline</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">🎯</div>
          <div class="stat-label">Motion</div>
          <div class="stat-value" id="stat-motion" style="font-size:1.1rem;margin-top:8px;">—</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">📋</div>
          <div class="stat-label">Events Today</div>
          <div class="stat-value" id="stat-events-today">—</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">📊</div>
          <div class="stat-label">Total Events</div>
          <div class="stat-value" id="stat-events-total">—</div>
        </div>
      </div>

      <div class="panel cam-wrap">
        <div class="cam-video-box" id="cam-box">
          <div class="cam-placeholder" id="cam-placeholder">🎥<br><br>Click <b>Start Camera</b> to begin<br>motion detection</div>
          <img id="cam-stream" style="display:none; width:100%; border-radius:10px;" alt="Camera feed">
          <div class="motion-indicator" id="motion-indicator">● Monitoring</div>
        </div>
        <div class="cam-controls">
          <button class="cam-btn cam-btn-green" onclick="startCamera()">▶ Start Camera</button>
          <button class="cam-btn cam-btn-red"   onclick="stopCamera()">■ Stop Camera</button>
        </div>
        <p id="cam-status"></p>
      </div>
    </div>

    <!-- ── View: Motion Events ── -->
    <div class="view" id="view-motion">
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">🎯 Motion Detection Log</span>
          <div style="display:flex;gap:0.5rem;">
            <button class="refresh-btn" onclick="loadMotionEvents()">↻ Refresh</button>
          </div>
        </div>
        <div id="motion-log-container"><div class="no-logs">Loading…</div></div>
      </div>
    </div>

    <!-- ── View: Login Logs ── -->
    <div class="view" id="view-logs">
      <div class="stats-row">
        <div class="stat-card"><div class="stat-icon">🔢</div><div class="stat-label">Total Logins</div><div class="stat-value" id="stat-total">—</div></div>
        <div class="stat-card"><div class="stat-icon">✅</div><div class="stat-label">Successful</div><div class="stat-value" id="stat-success" style="color:#059669;">—</div></div>
        <div class="stat-card"><div class="stat-icon">❌</div><div class="stat-label">Failed</div><div class="stat-value" id="stat-failed" style="color:#DC2626;">—</div></div>
        <div class="stat-card"><div class="stat-icon">👥</div><div class="stat-label">Unique Users</div><div class="stat-value" id="stat-unique">—</div></div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Login Activity</span>
          <button class="refresh-btn" onclick="loadLogs()">↻ Refresh</button>
        </div>
        <div class="filter-row">
          <button class="filter-btn active" id="filter-all"     onclick="setFilter('all')">All</button>
          <button class="filter-btn"        id="filter-success" onclick="setFilter('SUCCESS')">Success</button>
          <button class="filter-btn"        id="filter-failed"  onclick="setFilter('FAILED')">Failed</button>
        </div>
        <div id="log-container"><div class="no-logs">Loading…</div></div>
      </div>
    </div>

    <!-- ── View: Camera Settings ── -->
    <div class="view" id="view-settings">
      <div class="panel" style="max-width:600px;">
        <div class="panel-header">
          <span class="panel-title">⚙️ Network Camera Configuration</span>
        </div>
        <p style="font-size:0.88rem;color:#64748B;margin-bottom:1.2rem;line-height:1.6;">
          Connect a physical IP/CCTV camera over the network. Stop the camera stream before changing settings.
          Supported sources: local device index (<code>0</code>), RTSP streams, or HTTP MJPEG URLs.
        </p>

        <!-- Current config badge -->
        <div id="cfg-current" style="background:#F0F7FF;border:1px solid #BFDBFE;border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:1.2rem;font-size:0.85rem;color:#1E293B;">
          <span style="font-weight:700;color:#2563EB;">Active source:</span>
          <span id="cfg-active-source" style="font-family:monospace;">loading…</span>
          &nbsp;·&nbsp;
          <span id="cfg-active-label" style="color:#64748B;font-style:italic;"></span>
          <span style="float:right;font-size:0.75rem;color:#94A3B8;" id="cfg-updated-at"></span>
        </div>

        <!-- Input fields -->
        <div style="margin-bottom:0.9rem;">
          <label style="display:block;font-size:0.78rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">Camera Source</label>
          <input id="cfg-source" type="text" placeholder="e.g. rtsp://admin:pass@192.168.1.64:554/stream or 0"
            style="width:100%;padding:10px 14px;border:1.5px solid #BFDBFE;border-radius:10px;font-size:0.93rem;font-family:monospace;color:#1E293B;background:#F8FAFF;outline:none;"/>
          <div style="font-size:0.78rem;color:#94A3B8;margin-top:4px;">
            Common RTSP formats:&nbsp;
            <code>rtsp://user:pass@IP:554/stream1</code> &nbsp;·&nbsp;
            <code>http://IP:8080/video</code> &nbsp;·&nbsp;
            <code>0</code> for local webcam
          </div>
        </div>
        <div style="margin-bottom:1.2rem;">
          <label style="display:block;font-size:0.78rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">Camera Label (optional)</label>
          <input id="cfg-label" type="text" placeholder="e.g. Front Door, Parking Lot"
            style="width:100%;padding:10px 14px;border:1.5px solid #BFDBFE;border-radius:10px;font-size:0.93rem;color:#1E293B;background:#F8FAFF;outline:none;"/>
        </div>

        <!-- Tips -->
        <details style="margin-bottom:1.2rem;font-size:0.83rem;color:#475569;background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:0.8rem 1rem;">
          <summary style="cursor:pointer;font-weight:700;color:#92400E;">📡 Physical CCTV Setup Tips</summary>
          <ul style="margin-top:0.7rem;padding-left:1.2rem;line-height:1.9;">
            <li>Connect camera to router/switch via RJ-45 Ethernet or use PoE switch.</li>
            <li>Assign a <strong>static IP</strong> to the camera in your router's DHCP settings (or via camera's web UI).</li>
            <li>Default RTSP port is <strong>554</strong>; HTTP streams often use <strong>80</strong> or <strong>8080</strong>.</li>
            <li>Check camera manual for its stream path — common paths: <code>/stream1</code>, <code>/h264</code>, <code>/live/ch0</code>.</li>
            <li>Ensure the server running this app is on the <strong>same LAN</strong> or has routed access to the camera IP.</li>
            <li>Test with VLC → Media → Open Network Stream before configuring here.</li>
          </ul>
        </details>

        <div style="display:flex;gap:0.8rem;flex-wrap:wrap;">
          <button class="cam-btn" onclick="saveCameraConfig()" style="padding:10px 28px;">💾 Save Configuration</button>
          <button class="cam-btn" style="background:linear-gradient(135deg,#475569,#64748B);box-shadow:0 3px 10px rgba(71,85,105,0.3);padding:10px 20px;" onclick="loadCameraConfig()">↻ Reload</button>
        </div>
        <p id="cfg-msg" style="margin-top:0.9rem;font-size:0.85rem;min-height:1.2rem;font-weight:500;"></p>
      </div>
    </div>
</div><!-- .shell -->

<script>
  // ── Username ──
  const storedName = localStorage.getItem('yosan_user');
  if (storedName) {{
    document.getElementById('username').textContent = storedName;
    document.getElementById('user-avatar').textContent = storedName.charAt(0).toUpperCase();
  }}

  // ── View switching ──
  function showView(name) {{
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.getElementById('view-' + name).classList.add('active');
    document.getElementById('nav-' + name).classList.add('active');
    const titles = {{ camera: 'Camera', motion: 'Motion Events', logs: 'Login Logs', settings: 'Camera Settings' }};
    document.getElementById('page-title').textContent = titles[name] || name;
    if (name === 'logs')     loadLogs();
    if (name === 'motion')   loadMotionEvents();
    if (name === 'settings') loadCameraConfig();
  }}

  // ── Camera Config ──
  async function loadCameraConfig() {{
    try {{
      const res  = await fetch('/api/camera/config');
      const data = await res.json();
      document.getElementById('cfg-source').value       = data.source  || '';
      document.getElementById('cfg-label').value        = data.label   || '';
      document.getElementById('cfg-active-source').textContent = data.source  || '0';
      document.getElementById('cfg-active-label').textContent  = data.label   || '';
      document.getElementById('cfg-updated-at').textContent    = data.updated_at ? 'saved ' + data.updated_at : '';
    }} catch(e) {{
      document.getElementById('cfg-msg').textContent = '⚠️ Could not load config.';
      document.getElementById('cfg-msg').style.color = '#DC2626';
    }}
  }}

  async function saveCameraConfig() {{
    const source = document.getElementById('cfg-source').value.trim();
    const label  = document.getElementById('cfg-label').value.trim();
    const msg    = document.getElementById('cfg-msg');
    if (!source) {{ msg.textContent = '⚠️ Source is required.'; msg.style.color = '#DC2626'; return; }}
    msg.textContent = '⏳ Saving…'; msg.style.color = '#64748B';
    try {{
      const res  = await fetch('/api/camera/config', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ source, label }})
      }});
      const data = await res.json();
      if (res.ok) {{
        msg.textContent = '✅ Saved! Restart the camera stream to apply.';
        msg.style.color = '#059669';
        document.getElementById('cfg-active-source').textContent = data.source;
        document.getElementById('cfg-active-label').textContent  = data.label || '';
      }} else {{
        msg.textContent = '🔴 ' + (data.detail || 'Save failed.');
        msg.style.color = '#DC2626';
      }}
    }} catch(e) {{
      msg.textContent = '🔴 Server error: ' + e.message;
      msg.style.color = '#DC2626';
    }}
  }}

  // ── Camera ──
  let statusPollTimer = null;

  async function startCamera() {{
    const status = document.getElementById('cam-status');
    status.textContent = '⏳ Starting camera…';
    status.style.color = '#64748B';
    try {{
      // Tell backend to start OpenCV motion detection
      const res = await fetch('/api/camera/start', {{ method: 'POST' }});
      const data = await res.json();
      if (res.ok) {{
        // Point the img tag at the backend MJPEG stream (carries bounding boxes)
        const img = document.getElementById('cam-stream');
        img.src = '/api/camera/stream';
        img.style.display = 'block';
        document.getElementById('cam-placeholder').style.display = 'none';
        document.getElementById('stat-cam-status').textContent = 'Live';
        document.getElementById('stat-cam-status').style.color = '#059669';
        const srcLabel = data.source_type === 'network' ? '🌐 Network CCTV' : '🖥 Local device';
        status.textContent = `🟢 Camera active — ${{srcLabel}}`;
        status.style.color = '#059669';

        // Show motion indicator overlay and start polling
        document.getElementById('motion-indicator').classList.add('visible');
        startMotionPoll();
      }} else {{
        status.textContent = '🔴 ' + (data.detail || 'Could not start camera');
        status.style.color = '#DC2626';
      }}
    }} catch(e) {{
      status.textContent = '🔴 Server error: ' + e.message;
      status.style.color = '#DC2626';
    }}
  }}

  function startMotionPoll() {{
    if (statusPollTimer) clearInterval(statusPollTimer);
    statusPollTimer = setInterval(async () => {{
      try {{
        const r = await fetch('/api/camera/status');
        const d = await r.json();
        const ind  = document.getElementById('motion-indicator');
        const stat = document.getElementById('stat-motion');
        const camStatus = document.getElementById('cam-status');

        if (!d.active) {{ stopCamera(); return; }}

        // Show camera_error if the thread hit a problem
        if (d.error) {{
          camStatus.textContent = '⚠️ ' + d.error;
          camStatus.style.color = '#DC2626';
        }}

        if (d.motion) {{
          ind.textContent = '⚠ MOTION DETECTED';
          ind.className = 'motion-indicator visible alert';
          stat.textContent = '⚠ Motion';
          stat.style.color = '#DC2626';
        }} else {{
          ind.textContent = '● Monitoring';
          ind.className = 'motion-indicator visible ok';
          stat.textContent = 'Clear';
          stat.style.color = '#059669';
        }}
      }} catch(e) {{}}
    }}, 500);
  }}

  async function stopCamera() {{
    if (statusPollTimer) {{ clearInterval(statusPollTimer); statusPollTimer = null; }}
    try {{ await fetch('/api/camera/stop', {{ method: 'POST' }}); }} catch(e) {{}}
    const img = document.getElementById('cam-stream');
    img.src = '';
    img.style.display = 'none';
    document.getElementById('cam-placeholder').style.display = 'block';
    document.getElementById('stat-cam-status').textContent = 'Offline';
    document.getElementById('stat-cam-status').style.color = '#DC2626';
    document.getElementById('cam-status').textContent = '⏹ Camera stopped';
    document.getElementById('cam-status').style.color = '#64748B';
    const ind = document.getElementById('motion-indicator');
    ind.className = 'motion-indicator';
    document.getElementById('stat-motion').textContent = '—';
    document.getElementById('stat-motion').style.color = '';
  }}

  // ── Motion Events ──
  async function loadMotionEvents() {{
    document.getElementById('motion-log-container').innerHTML = '<div class="no-logs">Loading…</div>';
    try {{
      const res  = await fetch('/api/motion/events?limit=200');
      const data = await res.json();
      const events = data.events || [];
      document.getElementById('stat-events-today').textContent = data.today_count ?? '—';
      document.getElementById('stat-events-total').textContent = data.total ?? '—';
      if (!events.length) {{
        document.getElementById('motion-log-container').innerHTML = '<div class="no-logs">No motion events recorded yet.</div>';
        return;
      }}
      const rows = events.map(e => `
        <tr>
          <td style="color:#94A3B8;font-size:0.8rem;">${{e.id}}</td>
          <td>${{escHtml(e.start_time)}}</td>
          <td>${{e.end_time ? escHtml(e.end_time) : '<span style="color:#bbb;">ongoing</span>'}}</td>
          <td>${{e.duration_seconds != null ? e.duration_seconds.toFixed(2) + 's' : '—'}}</td>
        </tr>`).join('');
      document.getElementById('motion-log-container').innerHTML = `
        <table class="log-table">
          <thead><tr><th>#</th><th>Start Time</th><th>End Time</th><th>Duration</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>`;
    }} catch(e) {{
      document.getElementById('motion-log-container').innerHTML = '<div class="no-logs">⚠️ Could not load motion events.</div>';
    }}
  }}

  // ── Login Logs ──
  let allLogs = [];
  let currentFilter = 'all';

  function setFilter(f) {{
    currentFilter = f;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('filter-' + (f === 'all' ? 'all' : f.toLowerCase())).classList.add('active');
    renderTable();
  }}

  async function loadLogs() {{
    document.getElementById('log-container').innerHTML = '<div class="no-logs">Loading…</div>';
    try {{
      const res  = await fetch('/api/admin/logs?limit=200');
      const data = await res.json();
      allLogs = data.logs || [];
      updateStats();
      renderTable();
    }} catch(e) {{
      document.getElementById('log-container').innerHTML = '<div class="no-logs">⚠️ Could not load logs.</div>';
    }}
  }}

  function updateStats() {{
    const success = allLogs.filter(l => l.status === 'SUCCESS').length;
    const failed  = allLogs.filter(l => l.status === 'FAILED').length;
    const unique  = new Set(allLogs.map(l => l.email)).size;
    document.getElementById('stat-total').textContent   = allLogs.length;
    document.getElementById('stat-success').textContent = success;
    document.getElementById('stat-failed').textContent  = failed;
    document.getElementById('stat-unique').textContent  = unique;
  }}

  function renderTable() {{
    const filtered = currentFilter === 'all' ? allLogs : allLogs.filter(l => l.status === currentFilter);
    if (!filtered.length) {{
      document.getElementById('log-container').innerHTML = '<div class="no-logs">No login records found.</div>';
      return;
    }}
    const rows = filtered.map(l => `
      <tr>
        <td style="color:#94A3B8;font-size:0.8rem;">${{l.id}}</td>
        <td>${{escHtml(l.email)}}</td>
        <td>${{l.name ? escHtml(l.name) : '<span style="color:#bbb;">—</span>'}}</td>
        <td><span class="badge ${{l.status === 'SUCCESS' ? 'badge-success' : 'badge-failed'}}">${{l.status}}</span></td>
        <td style="color:#64748B;">${{escHtml(l.ip || '—')}}</td>
        <td style="color:#94A3B8;font-size:0.8rem;">${{escHtml(l.logged_at || '—')}}</td>
      </tr>`).join('');
    document.getElementById('log-container').innerHTML = `
      <table class="log-table">
        <thead><tr><th>#</th><th>Email</th><th>Name</th><th>Status</th><th>IP Address</th><th>Time</th></tr></thead>
        <tbody>${{rows}}</tbody>
      </table>`;
  }}

  function escHtml(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }}

  // Boot — load initial event counts
  (async () => {{
    try {{
      const res = await fetch('/api/motion/events?limit=1');
      const data = await res.json();
      document.getElementById('stat-events-today').textContent = data.today_count ?? '0';
      document.getElementById('stat-events-total').textContent = data.total ?? '0';
    }} catch(e) {{}}
  }})();
</script>
</body>
</html>
"""

# ---------- Page routes ----------
@app.get("/", response_class=HTMLResponse)
def landing():
    return HTMLResponse(content=LANDING_HTML)

@app.get("/signup", response_class=HTMLResponse)
def signup_page():
    return HTMLResponse(content=SIGNUP_HTML)

@app.get("/login", response_class=HTMLResponse)
def login_page():
    return HTMLResponse(content=LOGIN_HTML)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    return HTMLResponse(content=DASHBOARD_HTML)

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)