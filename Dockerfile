# Track 1 - Synthetic Eye Movement & Virtual Camera Rendering (CPU Cycles)
# docker build -t eyes .
# docker run --rm -v "$PWD/out:/work/out" eyes --input left          -> videos/left.mp4
# docker run --rm -v "$PWD/out:/work/out" eyes --clockwise           -> renders/ + metadata.json
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates python3 \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6 libegl1 libdecor-0-0 \
    && rm -rf /var/lib/apt/lists/*
ARG BLENDER_VERSION=4.5.10
RUN curl -fsSL https://download.blender.org/release/Blender4.5/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
      | tar -xJ -C /opt && ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender
WORKDIR /work
COPY . /work
ENV BLENDER=/usr/local/bin/blender
ENTRYPOINT ["python3", "run.py"]
CMD ["--clockwise"]
