FROM ubuntu:24.04

ARG FLUTTER_VERSION=3.44.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/flutter/bin:/opt/flutter/bin/cache/dart-sdk/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    unzip \
    xz-utils \
    zip \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 \
    --branch "${FLUTTER_VERSION}" \
    https://github.com/flutter/flutter.git \
    /opt/flutter

RUN flutter config --no-analytics \
    && flutter precache --web

WORKDIR /workspace