from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import requests
import tempfile
import os
import base64
import yt_dlp
import shutil
import subprocess
import threading
import time

# ================== CONFIG ==================
MODEL_PATH = "cleanliness-11m-100.pt"
MODEL_URL = "https://github.com/mirteldisa01/cleanliness-nmsai/releases/download/v1.1.0/cleanliness-11m-100.pt"

DIRTY_CLASSES = {"dryleaves", "grass", "tree"}
CONF_THRESHOLD = 0.29

MAX_VIDEO_SECONDS = 10     # Max processing duration (CPU protection)
MAX_FRAMES = 10            # Hard frame cap (we only process first frames)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

app = FastAPI(title="Cleanliness Detection API")

# ================= GLOBAL MODEL =================
model = None
model_lock = threading.Lock()   # Thread-safe inference lock


# ================= STARTUP =================
@app.on_event("startup")
def load_model_once():
    """
    Load model once when FastAPI starts.
    Prevents reloading model per request.
    """
    global model

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")


# ================= REQUEST BODY =================
class VideoRequest(BaseModel):
    video_url: str


# ================= RESIZE HELPER =================
def resize_fit(frame, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


# ================= DOWNLOAD FUNCTIONS =================
def download_direct_video(url):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise HTTPException(400, "Failed to download video")

    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            tmp.write(chunk)

    tmp.close()
    return tmp.name


def download_youtube_video(url):
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "video.mp4")

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"YouTube download failed: {e}")

    if not os.path.exists(output_path):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "Downloaded YouTube video invalid")

    return output_path


def download_video(url):
    if "youtube.com" in url or "youtu.be" in url:
        return download_youtube_video(url)
    return download_direct_video(url)


# ================= FORCE CONVERT =================
def convert_to_mp4(input_path):
    output_path = input_path + "_fixed.mp4"

    command = [
        "ffmpeg",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "+faststart",
        "-y",
        output_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if not os.path.exists(output_path):
        raise HTTPException(400, "FFmpeg conversion failed")

    return output_path


# ================= CORE PROCESS =================
def process_frame_from_video(video_path):
    """
    Only read limited frames for CPU safety.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "OpenCV cannot open video")

    start_time = time.time()
    frame_count = 0
    selected_frame = None

    try:
        while cap.isOpened():

            if time.time() - start_time > MAX_VIDEO_SECONDS:
                break

            if frame_count >= MAX_FRAMES:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Just take first valid frame for cleanliness check
            selected_frame = frame
            break

    finally:
        cap.release()

    if selected_frame is None:
        raise HTTPException(400, "No valid frame extracted")

    return selected_frame


# ================= ENDPOINT =================
@app.post("/process-video")
def process_video(data: VideoRequest):

    if not data.video_url:
        raise HTTPException(400, "Video URL required")

    original_path = download_video(data.video_url)

    try:
        fixed_path = convert_to_mp4(original_path)
        frame = process_frame_from_video(fixed_path)

    finally:
        if os.path.exists(original_path):
            os.remove(original_path)

    # ===== THREAD-SAFE YOLO INFERENCE =====
    with model_lock:
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            nms=False,
            max_det=300,
            verbose=False
        )

    boxes = results[0].boxes
    dirty_detected = False
    detections = []

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            is_dirty = cls_name in DIRTY_CLASSES
            color = (0, 0, 255) if is_dirty else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "is_dirty": is_dirty
            })

            if is_dirty:
                dirty_detected = True

    # Cleanup converted file
    if os.path.exists(fixed_path):
        os.remove(fixed_path)

    status = "Dirty" if dirty_detected else "Clean"

    frame = resize_fit(frame)
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": status,
        "message": "Area needs cleaning" if dirty_detected else "Area is clean",
        "detections": detections,
        "image_base64": img_base64
    }