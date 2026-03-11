FROM nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    libopencv-dev python3-opencv \
    libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY src/ /app/src/
COPY static/ /app/static/
COPY models/ /app/models/

WORKDIR /app

EXPOSE 8630 8631 8632

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=60s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8630/health', timeout=2)"

CMD ["python3", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8630"]
