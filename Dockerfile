FROM python:3.11-slim

# Thư viện hệ thống cần cho audio/resample/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY streaming.py ./

# Cổng service (code của bạn chạy ở 9242)
EXPOSE 9241

# Quan trọng: chạy uvicorn trực tiếp để bind 0.0.0.0 (bỏ qua host=127.0.0.1 trong __main__)
CMD ["uvicorn", "streaming:app", "--host", "0.0.0.0", "--port", "9241", "--log-level", "info"]
