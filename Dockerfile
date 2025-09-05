FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/opt/cache

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg ca-certificates curl \
 && ln -s /usr/bin/python3.11 /usr/local/bin/python \
 && python -m pip install --upgrade pip \
 && mkdir -p ${XDG_CACHE_HOME} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install torch (CUDA) & Python deps first (stable layers)
RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# --- Preload faster-whisper model into cache (persists in image)
# This downloads model files into /opt/cache/whisper (because of XDG_CACHE_HOME)
RUN python - <<'PY'
from faster_whisper import WhisperModel
import os
print("Cache dir:", os.getenv("XDG_CACHE_HOME"))
# Preload medium model (CPU-only for container compatibility)
try:
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print("✅ Preloaded faster-whisper medium model (CPU)")
except Exception as e:
    print(f"❌ Model preload failed: {e}")
    # Try smaller model as fallback
    model = WhisperModel("base", device="cpu", compute_type="int8") 
    print("✅ Preloaded faster-whisper base model (CPU fallback)")
PY

# Verify (optional)
RUN ls -lh ${XDG_CACHE_HOME} || true

# --- Copy app code last so edits don’t bust the cached model layer
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]