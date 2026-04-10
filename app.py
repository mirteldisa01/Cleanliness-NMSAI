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

BASE_CONF = 0.05
SMALL_CONF = 0.3
LARGE_CONF = 0.1
AREA_THRESHOLD = 0.05

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

app = FastAPI(title="Cleanliness Detection API")

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

# ================= CLUSTER =================
def cluster_to_two_boxes(boxes):
    if len(boxes) == 0:
        return []

    centers = [(b[0] + b[2]) // 2 for b in boxes]
    threshold = np.median(centers)

    group1 = [boxes[i] for i in range(len(boxes)) if centers[i] < threshold]
    group2 = [boxes[i] for i in range(len(boxes)) if centers[i] >= threshold]

    def merge(group):
        if len(group) == 0:
            return None

        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        conf = sum(b[4] for b in group) / len(group)

        return (x1, y1, x2, y2, conf)

    result = []
    for g in [group1, group2]:
        m = merge(g)
        if m:
            result.append(m)

    return result

# ================= VIDEO =================
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

def convert_to_mp4(input_path):
    output = input_path + "_fixed.mp4"

    subprocess.run(
        ["ffmpeg", "-i", input_path, "-vcodec", "libx264", "-acodec", "aac", "-y", output],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return output

def process_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(400, "No frame")

    return frame

# ================= ENDPOINT =================
@app.post("/process-video")
def process_video(video_file: UploadFile = File(...)):

    original_path = save_uploaded_video(video_file)
    fixed_path = convert_to_mp4(original_path)

    try:
        frame = process_frame(fixed_path)
        h, w = frame.shape[:2]
        frame_area = w * h

        # ===== YOLO =====
        with model_lock:
            results = model.predict(
                frame,
                conf=BASE_CONF,
                nms=False,
                max_det=300,
                verbose=False
            )

        boxes = results[0].boxes

        # ===== FILTER ADAPTIF 🔥 =====
        filtered_boxes = []

        if boxes is not None:
            for box in boxes:
                cls_name = model.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])

                if cls_name not in DIRTY_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                area = (x2 - x1) * (y2 - y1)
                ratio = area / frame_area

                threshold = LARGE_CONF if ratio >= AREA_THRESHOLD else SMALL_CONF

                if conf >= threshold:
                    filtered_boxes.append((x1, y1, x2, y2, conf))

        # ===== CLUSTER =====
        merged_boxes = cluster_to_two_boxes(filtered_boxes)

        detections = []
        dirty_detected = len(merged_boxes) > 0

        # ===== DRAW =====
        for (x1, y1, x2, y2, conf) in merged_boxes:
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