# CUDA runtime with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# a couple of sane defaults + put model cache in a predictable place
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/opt/cache \
    TOKENIZERS_PARALLELISM=false

# system deps
# - python3.11 + headers so webrtcvad can build
# - ffmpeg for decoding when needed
# - build-essential for any native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg ca-certificates curl build-essential \
 && ln -s /usr/bin/python3.11 /usr/local/bin/python \
 && python -m pip install --upgrade pip setuptools wheel \
 && mkdir -p ${XDG_CACHE_HOME} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install torch that matches CUDA 12.1 first so layers cache well
RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# app deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# prefetch Whisper large-v3 weights into the image cache
# set at build time: --build-arg PRELOAD_WHISPER=0 to skip
ARG PRELOAD_WHISPER=1
RUN if [ "$PRELOAD_WHISPER" = "1" ]; then python - <<'PY'\nimport os, pathlib, urllib.request\ncache = pathlib.Path(os.environ.get("XDG_CACHE_HOME","/root/.cache")) / "whisper"\ncache.mkdir(parents=True, exist_ok=True)\ndst = cache / "large-v3.pt"\nurl = "https://openaipublic.blob.core.windows.net/whisper/models/large-v3.pt"\nif not dst.exists():\n    print("Downloading:", url)\n    urllib.request.urlretrieve(url, dst)\n    print("Saved to:", dst)\nelse:\n    print("Whisper model already present:", dst)\nPY\n; fi

# bring in the app last so small code edits don’t bust the heavy layers
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
