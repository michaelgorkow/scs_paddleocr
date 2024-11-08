FROM paddlepaddle/paddle:2.5.2-gpu-cuda12.0-cudnn8.9-trt8.6

RUN pip install --upgrade pip && pip install fastapi gunicorn uvicorn[standard] pymupdf snowflake-snowpark-python[pandas] pillow
RUN python3 -m pip install paddleocr 

WORKDIR /app
COPY app /app

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "1", "--timeout", "0", "webservice:app", "-k", "uvicorn.workers.UvicornWorker", "--log-level", "error"]
