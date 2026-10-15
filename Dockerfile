# syntax=docker/dockerfile:1
# Mizuoto in a container: one image with the server, its web UI and every engine.
#
#   docker build -t mizuoto .                                    # torch's CUDA 12.8 build (NVIDIA driver 570 or newer)
#   docker build -t mizuoto:cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
#   docker build -t mizuoto --build-arg GATED=1 .                # with Fish Speech and IndexTTS (licence-gated)
#   docker run --gpus all -p 7869:7869 -v mizuoto-data:/data -e MIZUOTO_SERVER__ACCESS_TOKEN=change-me mizuoto
#
# Every engine runs inside the server's own Python; models are downloaded into /data/models. torch's build is chosen
# here (TORCH_INDEX: cu126 … cu132, rocm6.4 … rocm7.2, cpu, from download.pytorch.org) and Mizuoto is installed over it.
# The licence-gated engines are built only with --build-arg GATED=1; their licences must still be accepted
# (Models › Engines, or mizuoto engine accept-licence). Every optional front end is in: GPT-SoVITS's Japanese
# (OpenJTalk, compiled into a wheel in the wheels stage) and Korean, and IndexTTS-2.5's Japanese. The image has no C
# compiler, so the optional compute.compile setting runs eagerly (it warns once).

ARG PYTHON=3.11

# -- the web UI ------------------------------------------------------------------------------------
FROM oven/bun:1 AS web
WORKDIR /src/mizuoto/webui
COPY mizuoto/webui/package.json mizuoto/webui/bun.lock ./
RUN bun install --frozen-lockfile
COPY mizuoto/webui/ ./
RUN bun run build

# -- the wheels: mizuoto, and the gated engines' distributions with GATED=1 ---------------------------
FROM python:${PYTHON}-slim AS wheels
ARG GATED=0
# A compiler and CMake for OpenJTalk (the version mizuoto[gptsovits-ja] pins), published as an sdist only; the image
# gets its wheel.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "uv>=0.9"
RUN pip wheel --no-cache-dir --no-deps "pyopenjtalk==0.4.1" -w /wheels
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY scripts/build_wheels.py scripts/
COPY packages/native packages/native
COPY mizuoto mizuoto
COPY --from=web /src/mizuoto/webui/dist mizuoto/webui/dist
RUN python scripts/build_wheels.py host --skip-web --out /wheels && \
    if [ "$GATED" = "1" ]; then python scripts/build_wheels.py native --out /wheels; fi

# -- the image -------------------------------------------------------------------------------------
FROM python:${PYTHON}-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH="torch>=2.8"
# No system packages: PyAV, torch and the text libraries ship their native code in their wheels; nothing calls ffmpeg
# or sox. torch first, from its index, so installing Mizuoto keeps that build.
RUN pip install --no-cache-dir --index-url "$TORCH_INDEX" "$TORCH"
# Then the wheels (mizuoto, the gated engines when built) with every optional front end; the rest from PyPI.
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    extras="mizuoto[gptsovits-ja,gptsovits-ko]"; \
    if ls /wheels/mizuoto_indextts-*.whl >/dev/null 2>&1; then extras="$extras mizuoto-indextts[ja]"; fi; \
    pip install --no-cache-dir --find-links /wheels /wheels/mizuoto*.whl $extras
ENV MIZUOTO_DATA_DIR=/data \
    MIZUOTO_SERVER__OPEN_BROWSER=false \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 7869
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7869/api/v1/app/health', timeout=4)"
CMD ["mizuoto", "webui", "--host", "0.0.0.0", "--port", "7869", "--no-open"]
