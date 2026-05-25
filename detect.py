import cv2
import time
import os
import threading
import smtplib
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# ✅ LOAD CREDENTIALS FROM .env FILE
# ─────────────────────────────────────────────
load_dotenv()
SENDER_EMAIL    = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL  = os.getenv("RECEIVER_EMAIL")

# Verify karo ke .env properly load hua
if not all([SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
    raise ValueError("❌ .env file mein EMAIL credentials missing hain!")

print(f"✅ Email config loaded — sending from: {SENDER_EMAIL}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MASK_MODEL      = "models/mask_best.pt"
GLOVES_MODEL    = "models/gloves_best.pt"
DEVICE          = "cpu"
CAPTURE_DIR     = "captures"
COOLDOWN_SEC    = 10
CONF_MASK       = 0.50
CONF_GLOVES     = 0.60
FACE_PADDING    = 0.35
VIDEO_DURATION  = 5
WRITE_FPS       = 15

os.makedirs(CAPTURE_DIR, exist_ok=True)

COLORS = {
    "with_mask"            : (0, 255, 0),
    "without_mask"         : (0, 0, 255),
    "mask_weared_incorrect": (0, 165, 255),
    "with_gloves"          : (255, 255, 0),
    "without_gloves"       : (255, 0, 255),
}

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
print("⏳ Loading models...")
mask_model   = YOLO(MASK_MODEL)
gloves_model = YOLO(GLOVES_MODEL)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("✅ Models loaded! Starting webcam...\n")


# ─────────────────────────────────────────────
# FACE ZONE HELPERS
# ─────────────────────────────────────────────
def get_face_zones(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    zones = []
    h, w  = frame.shape[:2]
    for (fx, fy, fw, fh) in faces:
        pad_x = int(fw * FACE_PADDING)
        pad_y = int(fh * FACE_PADDING)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)
        zones.append((x1, y1, x2, y2))
    return zones


def is_inside_face_zone(box, face_zones):
    bx1, by1, bx2, by2 = box
    b_cx = (bx1 + bx2) / 2
    b_cy = (by1 + by2) / 2
    for (fx1, fy1, fx2, fy2) in face_zones:
        if fx1 <= b_cx <= fx2 and fy1 <= b_cy <= fy2:
            return True
    return False


# ─────────────────────────────────────────────
# DRAW BOX
# ─────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# ─────────────────────────────────────────────
# SEND EMAIL
# ─────────────────────────────────────────────
def send_email(image_path, video_path, timestamp, violation_types):
    def _send():
        try:
            print("📧 Sending email...")
            msg            = MIMEMultipart()
            msg["From"]    = SENDER_EMAIL
            msg["To"]      = RECEIVER_EMAIL
            msg["Subject"] = f"Safety Violation Detected — {timestamp}"

            violations_str = " | ".join(violation_types)
            body = f"""
SAFETY VIOLATION ALERT

Time      : {timestamp}
Violations: {violations_str}

Please review the attached image and 5-second video clip.

— Mask & Gloves Detector System
"""
            msg.attach(MIMEText(body, "plain"))

            for path, filename in [
                (image_path, "violation.jpg"),
                (video_path, "violation_clip.mp4")
            ]:
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition",
                                    f"attachment; filename={filename}")
                    msg.attach(part)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

            print(f"✅ Email sent to {RECEIVER_EMAIL}")

        except Exception as e:
            print(f"❌ Email failed: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ─────────────────────────────────────────────
# MAIN WEBCAM LOOP
# ─────────────────────────────────────────────
def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    phase              = "idle"
    last_cycle_done    = 0
    recorder           = None
    record_start_time  = 0
    current_video_path = None
    current_image_path = None
    current_timestamp  = None
    current_violations = []

    print(f"📐 Resolution: {width}x{height}  |  Write FPS fixed at {WRITE_FPS}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame   = frame.copy()
        violation       = False
        violation_types = []

        face_zones = get_face_zones(display_frame)

        mask_results = mask_model.predict(
            display_frame, device=DEVICE, verbose=False, conf=CONF_MASK)
        for result in mask_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label  = mask_model.names[cls_id]
                conf   = float(box.conf[0])
                coords = list(map(int, box.xyxy[0]))
                draw_box(display_frame, *coords, label, conf,
                         COLORS.get(label, (255, 255, 255)))
                if label in ("without_mask", "mask_weared_incorrect"):
                    violation = True
                    if label not in violation_types:
                        violation_types.append(label)

        gloves_results = gloves_model.predict(
            display_frame, device=DEVICE, verbose=False, conf=CONF_GLOVES)
        for result in gloves_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label  = gloves_model.names[cls_id]
                conf   = float(box.conf[0])
                coords = list(map(int, box.xyxy[0]))
                if is_inside_face_zone(coords, face_zones):
                    continue
                draw_box(display_frame, *coords, label, conf,
                         COLORS.get(label, (255, 255, 255)))
                if label == "without_gloves":
                    violation = True
                    if label not in violation_types:
                        violation_types.append(label)

        now = time.time()

        # ── IDLE ──────────────────────────────
        if phase == "idle":
            if violation:
                timestamp  = time.strftime("%Y%m%d_%H%M%S")
                img_path   = os.path.join(CAPTURE_DIR, f"violation_{timestamp}.jpg")
                vid_path   = os.path.join(CAPTURE_DIR, f"violation_{timestamp}.mp4")

                cv2.imwrite(img_path, display_frame)
                print(f"\n📸 Snapshot saved → {img_path}")

                recorder = cv2.VideoWriter(
                    vid_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(WRITE_FPS),
                    (width, height)
                )

                record_start_time  = now
                current_video_path = vid_path
                current_image_path = img_path
                current_timestamp  = timestamp
                current_violations = violation_types.copy()
                phase              = "recording"
                print(f"🔴 Recording started → {vid_path}")

                flash = np.ones_like(display_frame, dtype=np.uint8) * 255
                cv2.addWeighted(flash, 0.35, display_frame, 0.65, 0, display_frame)

        # ── RECORDING ─────────────────────────
        elif phase == "recording":
            elapsed   = now - record_start_time
            remaining = max(0.0, VIDEO_DURATION - elapsed)

            if elapsed < VIDEO_DURATION:
                recorder.write(display_frame)

            progress = min(elapsed / VIDEO_DURATION, 3.0)
            bar_w    = int(width * progress)
            cv2.rectangle(display_frame,
                          (0, height - 10), (bar_w, height),
                          (0, 80, 255), -1)
            cv2.putText(display_frame,
                        f"REC  {remaining:.1f}s remaining",
                        (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

            if elapsed >= VIDEO_DURATION:
                recorder.release()
                recorder = None
                size_kb  = os.path.getsize(current_video_path) / 1024
                print(f"🎬 Video done → {current_video_path}  ({size_kb:.1f} KB)")

                send_email(
                    current_image_path,
                    current_video_path,
                    current_timestamp,
                    current_violations
                )

                last_cycle_done = now
                phase           = "cooldown"
                print(f"⏳ Cooldown started — {COOLDOWN_SEC}s before next cycle\n")

        # ── COOLDOWN ──────────────────────────
        elif phase == "cooldown":
            cd_left = max(0.0, COOLDOWN_SEC - (now - last_cycle_done))
            cv2.putText(display_frame,
                        f"Cooldown: {cd_left:.1f}s",
                        (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            if cd_left == 0:
                phase = "idle"
                print("✅ Cooldown done — detector active again\n")

        # ── BANNER ────────────────────────────
        if phase == "recording":
            banner_color = (180, 0, 0)
            banner_text  = "  RECORDING VIOLATION"
        elif violation:
            banner_color = (0, 0, 200)
            banner_text  = "  WARNING: VIOLATION DETECTED"
        elif phase == "cooldown":
            banner_color = (100, 100, 0)
            banner_text  = "  COOLDOWN WAITING"
        else:
            banner_color = (0, 160, 0)
            banner_text  = "  ALL CLEAR"

        cv2.rectangle(display_frame, (0, 0),
                      (width, 45), banner_color, -1)
        cv2.putText(display_frame, banner_text, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Mask & Gloves Detector  |  Q to quit", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if recorder:
        recorder.release()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detector stopped.")


if __name__ == "__main__":
    run_webcam()