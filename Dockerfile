# ── Backend Dockerfile ───────────────────────────────────────
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    ffmpeg curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements_arm.txt ./

# amd64 → CUDA torch via requirements.txt
# arm64 → CPU torch via requirements_arm.txt
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        pip install --no-cache-dir -r requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu124; \
    else \
        pip install --no-cache-dir -r requirements_arm.txt; \
    fi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]