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
    new_w, new_h = int(w * scale), int(h * scale)

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

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 100_000:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "Downloaded YouTube video is invalid")

    return output_path


def download_video(url):
    if "youtube.com" in url or "youtu.be" in url:
        return download_youtube_video(url)
    return download_direct_video(url)


# FORCE CONVERT TO H264 (OpenCV SAFE)
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

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 100_000:
        raise HTTPException(400, "FFmpeg conversion failed")

    return output_path


# ================= ENDPOINT =================
@app.post("/process-video")
def process_video(data: VideoRequest):

    original_path = download_video(data.video_url)

    try:
        fixed_path = convert_to_mp4(original_path)

        cap = cv2.VideoCapture(fixed_path)
        if not cap.isOpened():
            raise HTTPException(400, "OpenCV cannot open converted video")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(400, "Failed to read first frame")

    finally:
        # cleanup original
        if os.path.exists(original_path):
            os.remove(original_path)

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

    # cleanup converted file
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