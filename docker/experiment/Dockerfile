# Docker image
# https://hub.docker.com/layers/library/python/3.10.12-slim/images/sha256-0549f89d5387b0df640f136594a1db3a2aaa8cfbf351ec988d07325f8997b77a?context=explore
FROM python:3.12.10-slim

# https://zenn.dev/40_comeback_eng/articles/0d7f10e0ff7089
ENV PYTHONUNBUFFERED 1

ARG WORKING_DIRECTORY
WORKDIR /${WORKING_DIRECTORY}

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt update && apt install -y --no-install-recommends \
    # For development and debug (procps: `free -h`)
    curl git wget net-tools vim procps unzip \
    # Necessary for pip install
    gcc g++ libc-dev \
    \
    bzip2 \
    build-essential \
    # Required for building bayes3d
    ninja-build \
    ca-certificates \
    # For cv2 / grad-cam
    libopencv-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

# NOTE: ../..から実行すること
# ADD . /${WORKING_DIRECTORY}/

# ----------------------------------------------------------------------
# Prepare uv
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
# ----------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Prepare packages from pyproject.toml
COPY docker/experiment/pyproject.toml /${WORKING_DIRECTORY}/pyproject.toml
# https://www.reddit.com/r/learnpython/comments/1hspt62/comment/m5sak3o/?tl=ja
ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
RUN uv sync


# RUN git clone https://github.com/probcomp/bayes3d.git
# WORKDIR /bayes3d
# # CUDA無効化
# RUN sed -i 's/CUDAExtension/CppExtension/g' setup.py
# RUN uv pip install -e .

# RUN git clone https://github.com/probcomp/bayes3d.git
# RUN cd bayes3d && \
#     # CUDA無効化
#     sed -i 's/CUDAExtension/CppExtension/g' setup.py && \
#     # fix typo
#     sed -i 's/jupterlab/jupyterlab/g' requirements-dev.txt && \
#     uv pip install -e . --system  --no-build-isolation