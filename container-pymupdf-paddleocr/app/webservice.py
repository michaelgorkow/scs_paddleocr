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

# Access the logger dictionary through the logging manager and set each to ERROR level
for logger_name in logging.Logger.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logging.getLogger('logger-pymupdf-paddleocr-core').setLevel(logging.INFO)
logger = logging.getLogger('logger-pymupdf-paddleocr-webservice')
logger.setLevel(logging.INFO)

total_documents_processed = 0
jobs = {}
app = FastAPI()
lock = asyncio.Lock()
job_queue = asyncio.Queue()

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
    
def download_file(document_file, retries=10, delay=2):
    attempt = 0
    start = time.time()
    while attempt < retries:
        try:
            response = requests.get(document_file, verify=False)
            response.raise_for_status()  # Raises an error for failed requests
            logger.debug(f'DOWNLOAD_TIME: {round(time.time() - start, 3)} # DOC: {document_file}')
            return response.content
        except requests.exceptions.RequestException as e:
            attempt += 1
            logger.error(f'Attempt {attempt} of {retries} - ERROR: Failed to download {document_file}. Error: {e}')
            time.sleep(delay)  # Delay before retrying
    logger.error(f'ERROR: Failed to download {document_file} after {retries} attempts')
    return None  # Return None if download failed after all retries

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
        start = time.time()
        # Process each document
        logger.debug(f'DOC:{relative_path}')
        relative_path = escape_path(relative_path)
        sql_string = f"SELECT GET_PRESIGNED_URL('{stage_name}', '{relative_path}')"
        result = conn.cursor().execute(sql_string).fetchall()
        document_file = result[0][0]
        document_file = download_file(document_file)
        _, file_extension = os.path.splitext(relative_path)
        file_extension = file_extension.lower()
        
        # PDF extraction
        if file_extension == '.pdf':
            extraction_results = extract_pdf(document_file)
        else:
            extraction_results = [f'ERROR: Received: {relative_path}']
        total_documents_processed += 1
        return_data.append([index, extraction_results])
        logger.info(f'EXTRACTION_TIME: {round(time.time() - start, 1)} # DOC: {relative_path}')

    # Update job status after processing
    async with lock:
        jobs[batch_id]['result'] = return_data
        jobs[batch_id]['status'] = 'DONE'
    
    logger.debug(f'Job {batch_id} completed.')

# Handler to receive requests
@app.post("/pymupdf-paddleocr-extract", tags=["Endpoints"], status_code=HTTPStatus.ACCEPTED)
async def handler(request: Request):
    global jobs
    batch_id = request.headers['sf-external-function-query-batch-id']
    logger.debug(f'Received Request: {batch_id}')
    request_body = await request.json()
    request_body = request_body['data']
    jobs[batch_id] = {'status': 'QUEUED'}
    await job_queue.put((batch_id, request_body))
    logger.debug(f'Job {batch_id} added to the queue.')
    return {"status": "QUEUED", "batch_id": batch_id}

@app.get("/pymupdf-paddleocr-extract", tags=["Endpoints"])
def get_results(request: Request, response: Response):
    global jobs
    batch_id = request.headers['sf-external-function-query-batch-id']
    logger.debug(f'Checking Status: {batch_id}')
    logger.debug(f'Number of Jobs in Queue: {len(jobs)}')
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
