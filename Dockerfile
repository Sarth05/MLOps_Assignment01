FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY models ./models
COPY reports ./reports

RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -e .
RUN python -m heart_ml.data --output data/raw/heart.csv \
    && python -m heart_ml.train --data data/raw/heart.csv --output models --reports reports/model

EXPOSE 8000
CMD ["uvicorn", "heart_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
