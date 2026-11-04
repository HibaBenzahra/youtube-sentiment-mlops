FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY --chown=user app_api.py .
COPY --chown=user src/api ./src/api
COPY --chown=user src/utils ./src/utils
COPY --chown=user models/model.joblib models/vectorizer.joblib ./models/

USER user


ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port ${PORT}"]

