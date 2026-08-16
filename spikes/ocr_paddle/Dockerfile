FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libglib2.0-0 \
       libgl1 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install \
       paddlepaddle==3.2.2 \
       paddleocr

WORKDIR /data