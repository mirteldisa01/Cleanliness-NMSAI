from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from ultralytics import YOLO
import cv2
import numpy as np
import requests
import tempfile
import os
import base64
import subprocess
import threading
import time
from urllib.parse import urlparse

# ================== CONFIG ==================
MODEL_PATH = "cleanliness-11x-100.pt"
MODEL_URL = "https://github.com/mirteldisa01/cleanliness-nmsai/releases/download/v1.2.0/cleanliness-11x-100.pt"

DIRTY_CLASSES = {"dryleaves", "grass", "tree"}
CONF_THRESHOLD = 0.29

MAX_VIDEO_SECONDS = 10     # Max processing duration (CPU protection)
MAX_FRAMES = 10            # Hard frame cap (we only process first frames)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"
}

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


# ================= URL / FILE VALIDATION =================
def is_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return "youtube.com" in host or "youtu.be" in host
    except Exception:
        return False


def has_valid_video_extension(url: str) -> bool:
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in ALLOWED_VIDEO_EXTENSIONS)
    except Exception:
        return False


# ================= DOWNLOAD / SAVE FUNCTIONS =================
def download_direct_video(url: str):
    """
    Download ONLY direct video URLs (e.g. .mp4, .mov, etc.)
    No YouTube support.
    """
    if is_youtube_url(url):
        raise HTTPException(
            status_code=400,
            detail=(
                "YouTube URLs are no longer supported in this endpoint. "
                "Please upload the video file directly or provide a direct video file URL."
            )
        )

    # Optional early validation based on URL extension
    if not has_valid_video_extension(url):
        # We still allow download attempt because some valid URLs don't expose extension clearly
        pass

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                raise HTTPException(400, "Failed to download video")

            content_type = r.headers.get("Content-Type", "").lower()
            if content_type and not content_type.startswith("video/"):
                # Not all servers return perfect content-type, but this is a useful safeguard
                # We'll still allow octet-stream because some file servers use it.
                if "application/octet-stream" not in content_type:
                    raise HTTPException(
                        400,
                        f"URL does not appear to be a direct video file (Content-Type: {content_type})"
                    )

            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)

        tmp.close()

        if os.path.getsize(tmp_path) == 0:
            raise HTTPException(400, "Downloaded video file is empty")

        return tmp_path

    except HTTPException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(400, f"Failed to download direct video: {str(e)}")


def save_uploaded_video(upload_file: UploadFile):
    """
    Save uploaded video file to temp storage.
    """
    if upload_file is None:
        raise HTTPException(400, "Video file is required")

    filename = (upload_file.filename or "").lower()
    ext = os.path.splitext(filename)[1]

    if ext and ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported video file extension: {ext}. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext if ext else ".mp4")
    tmp_path = tmp.name

    try:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

        tmp.close()

        if os.path.getsize(tmp_path) == 0:
            raise HTTPException(400, "Uploaded video file is empty")

        return tmp_path

    except HTTPException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(400, f"Failed to save uploaded video: {str(e)}")
    finally:
        try:
            upload_file.file.close()
        except Exception:
            pass


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
def process_video(
    video_url: str = Form(None),
    video_file: UploadFile = File(None)
):
    """
    Process video from:
    1. uploaded file (video_file), OR
    2. direct video file URL (video_url)

    NOTE:
    - YouTube URLs are NOT supported.
    - Output format remains unchanged.
    """

    if not video_url and not video_file:
        raise HTTPException(
            400,
            "Either 'video_file' upload or 'video_url' direct video link is required"
        )

    if video_url and video_file:
        raise HTTPException(
            400,
            "Please provide only one input: either 'video_file' or 'video_url', not both"
        )

    original_path = None
    fixed_path = None

    try:
        if video_file:
            original_path = save_uploaded_video(video_file)
        else:
            original_path = download_direct_video(video_url)

        fixed_path = convert_to_mp4(original_path)
        frame = process_frame_from_video(fixed_path)

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

    finally:
        # Cleanup files
        if original_path and os.path.exists(original_path):
            os.remove(original_path)

        if fixed_path and os.path.exists(fixed_path):
            os.remove(fixed_path)