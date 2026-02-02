import streamlit as st
import cv2
import requests
import numpy as np
import os
import tempfile
from ultralytics import YOLO

# ================= CONFIG =================
st.set_page_config(
    page_title="🧹 Cleanliness Detection",
    layout="wide"
)

MODEL_PATH = "https://github.com/mirteldisa01/Cleanliness-NMSAI/releases/download/v1.0/cleanliness-x-100.pt"
DIRTY_CLASSES = {"dryleaves", "grass", "tree"}
CONF_THRESHOLD = 0.29

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

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

def download_video(url):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".video")
    r = requests.get(url, stream=True, timeout=30)
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        tmp.write(chunk)
    tmp.close()
    return tmp.name

# ================= UI =================
st.title("🧹 Cleanliness Detection")

st.markdown("Enter Video URL(**.mp4 / .webm**)")

video_url = st.text_input(
    "Enter Video URL (.mp4 / .webm)",
    placeholder="http://example.com/video.webm"
)

if st.button("Process Video") and video_url:
    with st.spinner("Processing video..."):

        # ===== Download video =====
        video_path = download_video(video_url)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        os.remove(video_path)

        if not ret:
            st.error("The video cannot be opened. Please make sure the URL is valid and publicly accessible.")
            st.stop()

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

                if is_dirty:
                    dirty_detected = True

        status = "Dirty" if dirty_detected else "Clean"

        frame = resize_fit(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ================= OUTPUT (SS2 STYLE) =================
    st.subheader(f"Final Status: {status}")

    if dirty_detected:
        st.info("Final Status: Area needs to be cleaned.")
    else:
        st.info("Final Status: Area is clean.")


    left, center, right = st.columns([30, 40, 30])
    with center:
        st.image(frame, use_container_width=True)
