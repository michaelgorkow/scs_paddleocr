import logging
import sys
from fastapi import FastAPI, Request, BackgroundTasks, Response
from http import HTTPStatus
import requests
import os
from snowflake.connector import connect
import asyncio
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from extraction.core import extract_pdf

total_documents_processed = 0
jobs = {}
app = FastAPI()
lock = asyncio.Lock()
job_queue = asyncio.Queue()

# Logging
def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
        '%(name)s [%(asctime)s] [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    return logger

logger = get_logger('logger-pymupdf-paddleocr-webservice')

def get_login_token():
    """Read the login token supplied automatically by Snowflake."""
    with open("/snowflake/session/token", "r") as f:
        return f.read()
  
def escape_path(path: str):
    """Escape paths containing quotes to prevent SQL errors."""
    return path.replace("'", "\\'")

async def process_queue():
    """Processes jobs in the queue one at a time."""
    while True:
        batch_id, request_body = await job_queue.get()
        try:
            await run_extraction(batch_id, request_body)
        finally:
            job_queue.task_done()

async def run_extraction(batch_id, request_body):
    global jobs
    global total_documents_processed
    async with lock:
        jobs[batch_id] = {'status': 'RUNNING'}

    # Processing code here (no lock needed for reading/using jobs in this part)
    return_data = []
    conn = connect(
        host=os.getenv('SNOWFLAKE_HOST'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        token=get_login_token(),
        authenticator='oauth'
    )

    for index, stage_name, relative_path in request_body:
        # Process each document
        logger.info(f'DOCUMENT_PATH:{relative_path}')
        relative_path = escape_path(relative_path)
        sql_string = f"SELECT GET_PRESIGNED_URL('{stage_name}', '{relative_path}')"
        result = conn.cursor().execute(sql_string).fetchall()
        document_file = result[0][0]
        document_file = requests.get(document_file, verify=False).content
        _, file_extension = os.path.splitext(relative_path)
        file_extension = file_extension.lower()
        
        # PDF extraction
        if file_extension == '.pdf':
            extraction_results = extract_pdf(document_file)
        total_documents_processed += 1
        return_data.append([index, extraction_results])

    # Update job status after processing
    async with lock:
        jobs[batch_id]['result'] = return_data
        jobs[batch_id]['status'] = 'DONE'
    
    logger.info(f'Job {batch_id} completed.')

# Handler to receive requests
@app.post("/pymupdf-paddleocr-extract", tags=["Endpoints"], status_code=HTTPStatus.ACCEPTED)
async def handler(request: Request):
    global jobs
    batch_id = request.headers['sf-external-function-query-batch-id']
    logger.info(f'Received Request: {batch_id}')
    request_body = await request.json()
    request_body = request_body['data']
    jobs[batch_id] = {'status': 'QUEUED'}
    await job_queue.put((batch_id, request_body))
    logger.info(f'Job {batch_id} added to the queue.')
    return {"status": "QUEUED", "batch_id": batch_id}

@app.get("/pymupdf-paddleocr-extract", tags=["Endpoints"])
def get_results(request: Request, response: Response):
    global jobs
    batch_id = request.headers['sf-external-function-query-batch-id']
    logger.info(f'Checking Status: {batch_id}')
    if batch_id not in jobs:
        response.status_code = HTTPStatus.NOT_FOUND
        return {"data": [], "error": "Job not found"}
    batch_status = jobs[batch_id]['status']
    if batch_status == 'RUNNING' or batch_status == 'QUEUED':
        response.status_code = HTTPStatus.ACCEPTED
        return {"data": []}
    elif batch_status == 'DONE':
        return_data = jobs[batch_id]['result']
        del jobs[batch_id]
        return {"data": return_data}
    else:
        logger.error("Unknown job status")
        response.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        return {"data": [], "error": "Unknown job status"}
        
# Start the queue processor
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())
