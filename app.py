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

# ================= CONFIG =================
MODEL_PATH = "cleanliness-11x-100.pt"
MODEL_URL = "https://github.com/mirteldisa01/cleanliness-nmsai/releases/download/v1.2.0/cleanliness-11x-100.pt"

DIRTY_CLASSES = {"dryleaves", "grass", "tree"}

CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
MAX_DET = 300
FRAME_SKIP = 90
FPS = 30

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

app = FastAPI(
    title="Cleanliness Detection API",
    version="1.2.0"
)

model = None
model_lock = threading.Lock()

# ================= STARTUP =================
@app.on_event("startup")
def load_model_once():
    global model

    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = YOLO(MODEL_PATH)
    print("Model loaded")


# ================= IOU =================
def compute_iou(box1, box2):
    x1, y1, x2, y2, _ = box1
    x1g, y1g, x2g, y2g, _ = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# ================= NMS =================
def non_max_suppression(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    selected = []

    while boxes:
        best = boxes.pop(0)
        selected.append(best)

        boxes = [
            box for box in boxes
            if compute_iou(best, box) < iou_threshold
        ]

    return selected


# ================= RESIZE =================
def resize_fit(frame):
    h, w = frame.shape[:2]
    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    x_offset = (TARGET_WIDTH - new_w) // 2
    y_offset = (TARGET_HEIGHT - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


# ================= VIDEO HANDLER =================
def save_uploaded_video(file: UploadFile):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    path = tmp.name

    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        tmp.write(chunk)

    tmp.close()
    file.file.close()
    return path


def download_video_from_url(url: str):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    path = tmp.name

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise HTTPException(400, "Failed to download video")

    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            tmp.write(chunk)

    tmp.close()
    return path


def convert_to_mp4(input_path):
    output = input_path + "_fixed.mp4"

    subprocess.run(
        ["ffmpeg", "-i", input_path, "-vcodec", "libx264", "-acodec", "aac", "-y", output],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return output


# ================= FRAME EXTRACTION =================
def process_frame(video_path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.close()

    t = FRAME_SKIP / float(FPS)

    cmd = [
        "ffmpeg",
        "-ss", str(t),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        tmp.name
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame = cv2.imread(tmp.name)
    os.remove(tmp.name)

    if frame is None:
        raise HTTPException(400, "Failed to extract frame")

    return frame


# ================= ENDPOINT =================
@app.post("/process-video")
def process_video(
    video_file: UploadFile = File(None),
    video_url: str = Form(None)
):
    if not video_file and not video_url:
        raise HTTPException(400, "Provide video_file or video_url")

    if video_file:
        original_path = save_uploaded_video(video_file)
    else:
        original_path = download_video_from_url(video_url)

    fixed_path = convert_to_mp4(original_path)

    try:
        frame = process_frame(fixed_path)

        # ===== SAVE LAST FRAME (FOR FALLBACK) =====
        last_frame = frame.copy()

        # ===== YOLO =====
        with model_lock:
            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                nms=False,
                max_det=MAX_DET,
                verbose=False
            )

        boxes = results[0].boxes

        # ===== FILTER DIRTY =====
        filtered_boxes = []

        if boxes is not None:
            for box in boxes:
                cls_name = model.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])

                if cls_name in DIRTY_CLASSES and conf >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    filtered_boxes.append((x1, y1, x2, y2, conf))

        # ===== APPLY NMS =====
        final_boxes = non_max_suppression(filtered_boxes, IOU_THRESHOLD)

        # ===== OUTPUT =====
        detections = []
        dirty_detected = len(final_boxes) > 0

        for (x1, y1, x2, y2, conf) in final_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.putText(
                frame,
                f"Dirty {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            detections.append({
                "class": "dirty_area",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "is_dirty": True
            })

        # ===== STATUS =====
        status = "Dirty" if dirty_detected else "Clean"

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if dirty_detected else (0, 255, 0),
            3
        )

        # ===== FALLBACK: NO DIRTY DETECTED =====
        if not dirty_detected and last_frame is not None:
            fallback_frame = last_frame.copy()

            h, w = fallback_frame.shape[:2]

            cv2.putText(
                fallback_frame,
                "CLEAR",
                (w - 200, 40),  # kanan atas
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),  # hijau
                3
            )

            frame = fallback_frame

        # ===== FINAL OUTPUT =====
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
        os.remove(original_path)
        os.remove(fixed_path)
