FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/app

# Copy MedVAE first for editable install
COPY model/ /opt/app/model/

# Copy requirements and install deps
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# Copy the rest of the source
COPY . /opt/app

# Environment variables (no host dirs)
ENV PYTHONUNBUFFERED=1 \
    TMP_DIR=/tmp \
    OUTPUT_DIR=/tmp/output

# Ensure output dir exists
RUN mkdir -p /tmp/output

# Default command
CMD ["python", "run.py"]
