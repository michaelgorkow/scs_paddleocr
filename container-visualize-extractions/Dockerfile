FROM python:3.10.13-slim-bullseye

RUN pip install --upgrade pip && \
    pip install fastapi gunicorn uvicorn[standard] opencv-python-headless==4.5.4.58 snowflake-snowpark-python pymupdf Pillow

WORKDIR /app
COPY app /app

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "1", "--timeout", "0", "webservice:app", "-k", "uvicorn.workers.UvicornWorker"]
