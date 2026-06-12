# AutoSlicer inference image
#   build : docker build -t autoslicer .
#   run   : docker run --rm --gpus all -v "$PWD":/work autoslicer \
#             --input "https://www.bilibili.com/video/BVxxxxxxxxxx" \
#             --checkpoint /work/best.pt --out-dir /work/out --cut-video
#   no GPU: drop --gpus all and add --device cpu
# Model weights are not baked in — download best.pt from the Releases page
# and mount it (keeps the image generic across streamers/models).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml LICENSE README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

WORKDIR /work
ENTRYPOINT ["autoslicer"]
CMD ["--help"]
