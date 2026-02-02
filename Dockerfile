FROM python:3.10-slim

# ===== System deps (opencv & ffmpeg) =====
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ===== Working directory =====
WORKDIR /app

# ===== Copy requirements =====
COPY requirements.txt .

# ===== Install Python deps =====
RUN pip install --no-cache-dir -r requirements.txt

# ===== Copy app =====
COPY . .

# ===== Streamlit config =====
EXPOSE 8001

# ===== Run Streamlit =====
CMD ["streamlit", "run", "app.py", "--server.port=8001", "--server.address=0.0.0.0"]
