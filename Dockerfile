# syntax=docker/dockerfile:1
#
# BUILT AND VERIFIED. `docker build` runs clean against this file:
#   docker build -t finhm/ml1-governed-fraud:latest .
# Image size 282MB, Docker Engine 29.1.3 on Ubuntu 26.04 (WSL2).
#
# AND RUNNING IT FOUND THE IMAGE WAS BROKEN. `docker build` succeeded and
# `docker run` died on startup:
#
#   OSError: libgomp.so.1: cannot open shared object file
#
# lightgbm links against libgomp (the GNU OpenMP runtime) and python:3.12-slim
# does not ship it. A pip install of a manylinux wheel resolves and installs
# perfectly; the shared library it needs at LOAD time is a system package pip
# knows nothing about. So the build was green and the service could not start,
# for as long as nobody ran it.
#
# That is the whole argument for running a container rather than building one:
# `docker build` proves the layers resolve and proves nothing about whether the
# thing inside starts.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: a source change must not invalidate this layer.
# libgomp1: lightgbm's OpenMP runtime. NOT optional and NOT a pip dependency --
# the wheel installs without it and fails at import. Installed before pip so a
# requirements change does not re-run apt.
RUN apt-get update     && apt-get install -y --no-install-recommends libgomp1     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root. A root container that mounts a volume writes root-owned files onto
# the host, which is somebody else's afternoon.
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

