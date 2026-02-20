# ==============================
# Base Image
# ==============================
FROM python:3.10-slim

# ==============================
# Environment Variables
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# System Dependencies (OpenCV)
# ==============================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# Working Directory
# ==============================
WORKDIR /app

# ==============================
# Install Python Dependencies
# ==============================
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ==============================
# Copy Application Code
# ==============================
COPY . .

# ==============================
# Expose FastAPI Port
# ==============================
EXPOSE 8004

# ==============================
# Run FastAPI (Uvicorn)
# ==============================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]