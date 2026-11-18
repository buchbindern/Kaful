# syntax=docker/dockerfile:1
# Kaful: the deployment unit (target `runtime`) and the image CI runs the offline tests in (target `test`). M15.
#
#   docker build --target runtime -t kaful .
#   docker run --rm --env-file .env -v "$PWD/machines:/app/machines" kaful run eversys_coffee_machine --budget 1.0
#   docker build --target test -t kaful-test . && docker run --rm --network none kaful-test
#
# kaful reads its config, prompts, eval results and baselines relative to the folder holding the kaful/ package
# (kaful.config.REPO_ROOT), so the source tree is the app (/app on PYTHONPATH), not a site-packages install.
# Machine documents, runs and caches are never baked in: mount machines/ at run time.

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS wheels

# Some dependencies (numpoly, via progpy -> chaospy) ship no wheel for every platform: build them here, so the
# runtime image installs wheels only and carries no compiler.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt requirements-dev.txt constraints.txt /tmp/
RUN pip wheel --wheel-dir /wheels -r /tmp/requirements.txt -r /tmp/requirements-dev.txt -c /tmp/constraints.txt


FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 kaful
WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-index --find-links /wheels -r requirements.txt -c constraints.txt

COPY pyproject.toml kaful.toml pricing.toml ./
COPY kaful ./kaful
COPY evals ./evals
COPY docs/m1_baseline ./docs/m1_baseline
COPY docs/m1_cost_baseline.md ./docs/
COPY docs/legacy_baseline ./docs/legacy_baseline
COPY machines ./machines
RUN chown -R kaful:kaful /app

USER kaful
ENTRYPOINT ["python", "-B", "-m", "kaful"]
CMD ["--help"]


FROM runtime AS test

USER root
COPY requirements-dev.txt ./
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-index --find-links /wheels -r requirements-dev.txt -c constraints.txt
COPY tests ./tests
COPY scripts ./scripts
RUN chown -R kaful:kaful /app/tests /app/scripts

USER kaful
ENTRYPOINT []
CMD ["python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
