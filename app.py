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

# ================= CONFIG =================
MODEL_PATH = "https://github.com/mirteldisa01/Cleanliness-NMSAI/releases/download/v1.0/cleanliness-x-100.pt"
DIRTY_CLASSES = {"dryleaves", "grass", "tree"}
CONF_THRESHOLD = 0.29

app = FastAPI(title="Cleanliness Detection API")

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= REQUEST BODY =================
class VideoRequest(BaseModel):
    video_url: str

# ================= HELPER =================
def resize_fit(frame, target_w=1280, target_h=720):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def download_direct_video(url):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    r = requests.get(url, stream=True, timeout=60)

    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download video")

    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            tmp.write(chunk)

    tmp.close()
    return tmp.name


def download_youtube_video(url):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'merge_output_format': 'mp4',
        'outtmpl': tmp.name,
        'quiet': True,
        'noplaylist': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download YouTube video: {str(e)}")

    return tmp.name


def download_video(url):
    if "youtube.com" in url or "youtu.be" in url:
        return download_youtube_video(url)
    else:
        return download_direct_video(url)


# ================= ENDPOINT =================
@app.post("/process-video")
def process_video(data: VideoRequest):

    video_path = download_video(data.video_url)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    # Hapus file sementara
    if os.path.exists(video_path):
        os.remove(video_path)

    if not ret:
        raise HTTPException(
            status_code=400,
            detail="Video cannot be opened or invalid format"
        )

    # ===== YOLO Predict =====
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
                (x1, y2 + 20),
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

    status = "Dirty" if dirty_detected else "Clean"

    frame = resize_fit(frame)

    # Convert image to base64
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "status": status,
        "message": "Area needs cleaning" if dirty_detected else "Area is clean",
        "detections": detections,
        "image_base64": img_base64
    }