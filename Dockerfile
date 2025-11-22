FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR /app

# System deps for LightEval & compiling helpers
RUN apt-get update && apt-get install -y \
    git build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python", "--version"]