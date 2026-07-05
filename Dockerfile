# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — User Behavior Intelligence API
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t ubi-api .
# Run:    docker run -p 8001:8001 ubi-api
# Docs:   http://localhost:8001/docs
# ─────────────────────────────────────────────────────────────────────────────


# ── INSTRUCTION 1: FROM ───────────────────────────────────────────────────────
# Every Dockerfile starts with FROM — it picks the base image to build on top of.
# Think of it as "start with this pre-built environment".
#
# python:3.11-slim  means:
#   - Python 3.11 pre-installed
#   - "slim" = stripped-down Debian Linux, no docs/tests/extras
#   - Result: ~130MB base vs ~900MB for the full image
#
# Why 3.11 and not 3.14 (what Streamlit Cloud uses)?
# Stability. 3.14 is brand new and some packages don't support it yet.
# In production you pin to a version you've tested against.
FROM python:3.11-slim


# ── INSTRUCTION 2: WORKDIR ────────────────────────────────────────────────────
# Sets the working directory inside the container for all subsequent commands.
# If the directory doesn't exist, Docker creates it.
#
# /app is a convention — your code lives at /app inside the container.
# Every COPY, RUN, CMD after this runs relative to /app.
WORKDIR /app


# ── INSTRUCTION 3: Install system dependencies ────────────────────────────────
# Some Python packages need C libraries to compile.
# xgboost and shap need libgomp1 (OpenMP for parallel tree building).
#
# RUN executes a shell command during the BUILD phase (not at runtime).
# We chain with && and clean up with rm -rf to keep the layer small.
# Each RUN instruction creates a new image layer — fewer layers = smaller image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# ── INSTRUCTION 4: COPY requirements first (layer caching trick) ──────────────
# Docker builds images layer by layer. If a layer hasn't changed, Docker reuses
# the cached version — dramatically speeding up rebuilds.
#
# By copying ONLY requirements.txt first and installing packages BEFORE copying
# your code, you get this cache behaviour:
#   - Code changed, requirements same → reuse the pip install layer (fast)
#   - Requirements changed → reinstall packages, then copy code
#
# If you copied everything at once, any code change would invalidate the
# package install layer and force a full reinstall every time.
COPY requirements.txt .


# ── INSTRUCTION 5: Install Python packages ────────────────────────────────────
# --no-cache-dir  = don't store the pip download cache in the image (saves ~50MB)
# --upgrade pip   = avoid outdated pip warnings
RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt


# ── INSTRUCTION 6: COPY source code ──────────────────────────────────────────
# Now we copy the actual project files.
# This layer changes every time you edit code — but packages are already cached
# from the layer above, so rebuilds stay fast.
#
# We copy specific directories rather than everything (. -> .) to keep the
# image lean. The .dockerignore file also filters what gets copied.
COPY src/   ./src/
COPY api/   ./api/
COPY data/  ./data/


# ── INSTRUCTION 7: ENV ────────────────────────────────────────────────────────
# Set environment variables inside the container.
#
# PYTHONUNBUFFERED=1  — print() and logging output appear in docker logs
#                       immediately, not stuck in a buffer. Essential for
#                       seeing startup messages and errors in real time.
#
# PYTHONDONTWRITEBYTECODE=1 — don't create .pyc bytecode files.
#                             Saves a little disk space inside the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


# ── INSTRUCTION 8: EXPOSE ─────────────────────────────────────────────────────
# Documents which port the container listens on.
# This is DOCUMENTATION only — it doesn't actually open the port.
# The port is opened when you run: docker run -p 8001:8001
#   -p host_port:container_port
#   Left side (8001) = port on YOUR machine
#   Right side (8001) = port inside the container
EXPOSE 8001


# ── INSTRUCTION 9: CMD ────────────────────────────────────────────────────────
# The command that runs when the container STARTS (not during build).
# This is the difference between RUN (build time) and CMD (run time).
#
# We use JSON array form ["uvicorn", ...] rather than a string — this runs
# uvicorn directly without a shell wrapper, so signals (Ctrl+C, docker stop)
# reach the process correctly and shutdown is clean.
#
# --host 0.0.0.0  = listen on all network interfaces inside the container
#                   (not just localhost). Without this, the port is only
#                   reachable from inside the container itself.
# --port 8001     = the port the server binds to (matches EXPOSE above)
# --workers 1     = one worker process (fine for demos; scale up in production)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
