FROM debian:13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       tesseract-ocr-fra \
       tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /data

ENTRYPOINT ["tesseract"]
