# CUDA runtime with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/opt/cache \
    TOKENIZERS_PARALLELISM=false

# System deps:
# - python3.11 + headers (webrtcvad builds)
# - ffmpeg (if you use Opus->PCM path)
# - build-essential (native wheels)
# - curl (prefetch model)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg ca-certificates curl build-essential \
 && ln -s /usr/bin/python3.11 /usr/local/bin/python \
 && python -m pip install --upgrade pip setuptools wheel \
 && mkdir -p ${XDG_CACHE_HOME} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch (CUDA 12.1) first so layers cache well
RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Python deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Optional: prefetch Whisper large-v3 weights into the image
# Toggle with: --build-arg PRELOAD_WHISPER=0
ARG PRELOAD_WHISPER=1
RUN if [ "$PRELOAD_WHISPER" = "1" ]; then \
      mkdir -p "${XDG_CACHE_HOME}/whisper" && \
      if [ ! -f "${XDG_CACHE_HOME}/whisper/large-v3.pt" ]; then \
        echo "Prefetching Whisper large-v3 model..."; \
        curl -L -o "${XDG_CACHE_HOME}/whisper/large-v3.pt" \
          https://openaipublic.blob.core.windows.net/whisper/models/large-v3.pt; \
      else \
        echo "Whisper model already present."; \
      fi; \
    fi

# App code last so code edits don’t bust cached heavy layers
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
