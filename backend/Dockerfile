FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY pitchiq ./pitchiq
COPY alembic ./alembic
COPY alembic.ini ./
RUN pip install --no-cache-dir -e .
ENV PITCHIQ_DATA_DIR=/app/data PITCHIQ_ARTIFACT_DIR=/app/artifacts
