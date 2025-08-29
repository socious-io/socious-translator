FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/opt/cache \
    TOKENIZERS_PARALLELISM=false

# OS deps (ffmpeg for whisper; build-essential helps if a wheel needs compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg ca-certificates curl build-essential \
 && ln -s /usr/bin/python3.11 /usr/local/bin/python \
 && python -m pip install --upgrade pip \
 && mkdir -p ${XDG_CACHE_HOME} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch (CUDA 12.1) first so later layers cache nicely
RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# App deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# --- Prefetch Whisper model file without loading it into Torch ---
# This just downloads large-v3.pt into /opt/cache/whisper so runtime is instant.
RUN python - <<'PY'
import os, pathlib, urllib.request
cache = pathlib.Path(os.environ["XDG_CACHE_HOME"]) / "whisper"
cache.mkdir(parents=True, exist_ok=True)
dst = cache / "large-v3.pt"
if not dst.exists():
    url = "https://openaipublic.blob.core.windows.net/whisper/models/large-v3.pt"
    print("Downloading:", url)
    urllib.request.urlretrieve(url, dst)  # ~3GB
    print("Saved to:", dst)
else:
    print("Whisper model already present:", dst)
PY


RUN ls -lh ${XDG_CACHE_HOME}/whisper || true

# App code last
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
