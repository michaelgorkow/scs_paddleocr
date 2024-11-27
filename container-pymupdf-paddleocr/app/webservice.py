import logging
import sys
from fastapi import FastAPI, Request
import requests
import os
from snowflake.connector import connect
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from extraction.core import extract_pdf

total_documents_processed = 0
app = FastAPI()

# Access the logger dictionary through the logging manager and set each to ERROR level
for logger_name in logging.Logger.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logging.getLogger('logger-pymupdf-paddleocr-core').setLevel(logging.DEBUG)
logger = logging.getLogger('logger-pymupdf-paddleocr-webservice')
logger.setLevel(logging.DEBUG)


def get_login_token():
    """
    Read the login token supplied automatically by Snowflake. These tokens
    are short lived and should always be read right before creating any new connection.
    """
    with open("/snowflake/session/token", "r") as f:
        return f.read()


def escape_path(path: str):
    """
    Paths with ' inside the url lead to errors when we generate the presigned-url.
    Therefore we need to escape them for SQL generation.
    """
    path = path.replace("'", "\\'")
    return path
    

def download_file(document_file, retries=10, delay=2, timeout=(5, 30), chunk_size=4194304):
    """
    Downloads a file using streaming with retries and timeouts, returning the content.
    
    Args:
        document_file (str): URL of the file to download.
        retries (int): Number of retry attempts if the download fails.
        delay (int): Delay (in seconds) between retry attempts.
        timeout (tuple): (connection_timeout, read_timeout).
        chunk_size (int): Size of chunks to read in bytes.

    Returns:
        bytes: The content of the downloaded file, or None if download fails.
    """
    attempt = 0
    start = time.time()
    
    while attempt < retries:
        try:
            logger.debug(f'[ATTEMPT {attempt + 1} OF {retries}] Starting download for: {document_file}')
            with requests.get(document_file, verify=False, stream=True, timeout=timeout) as response:
                response.raise_for_status()  # Raises an error for failed requests
                
                # Accumulate content from the streamed chunks
                content = b""
                chunk_index = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    logger.debug(f'[CURRENT DOCUMENT: {document_file}] [CHUNK: {chunk_index}]')
                    if chunk:  # Only append non-empty chunks
                        content += chunk
                    chunk_index += 1
                
                logger.debug(f'[CURRENT DOCUMENT: {document_file}] [DOWNLOAD_TIME: {round(time.time() - start, 3)}]')
                return content  # Return the full content as bytes

        except requests.exceptions.RequestException as e:
            attempt += 1
            logger.error( f'[ERROR: ATTEMPT {attempt} OF {retries}] Failed to download {document_file}. Error: {e}')
            time.sleep(delay)  # Delay before retrying
    
    logger.error(f'[ERROR: Failed to download {document_file} after {retries} attempts]')
    return None  # Return None if download failed after all retries

# Create connection to Snowflake
conn = connect(
    host=os.getenv('SNOWFLAKE_HOST'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    token=get_login_token(),
    authenticator='oauth'
)
logger.debug('Created Snowflake Connection.')

@app.post("/pymupdf-paddleocr-extract", tags=["Endpoints"])
async def extract_content(request: Request):
    global total_documents_processed
    start_time = time.time()
    # stage_name: @MY_STAGE_NAME
    # relative_path: relative path
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    # create a snowflake session to retrieve file-urls while calling the extraction (avoid 1 hour limit of PRESIGNED_URLs)
    start = time.time()
    for index, stage_name, relative_path in request_body:
        extraction_results = ''
        logger.debug(f'[PROCESSING DOCUMENT: [{relative_path}]]')
        relative_path = escape_path(relative_path)
        sql_string = f"SELECT GET_PRESIGNED_URL('{stage_name}','{relative_path}')"
        result = conn.cursor().execute(sql_string).fetchall()
        presigned_url = result[0][0]
        logger.debug(f'[Downloading file: {presigned_url}]')
        document_file = download_file(presigned_url)
        logger.info(f'[Finished Downloading file: {presigned_url}]')
        if document_file is None:
            extraction_results = {'OCR_RESULTS':[], 'PAGE_ROTATIONS':[], 'ERROR_MESSAGE':'Download failed.'}
        if document_file is not None:
            _, file_extension = os.path.splitext(relative_path)
            file_extension = file_extension.lower()
            # pdf extraction
            if file_extension == '.pdf':
                extraction_results = extract_pdf(document_file, relative_path)
        total_documents_processed += 1
        return_data.append([index, extraction_results])
    logger.info(f'[CURRENT DOCUMENT: {relative_path}] [TOTAL_PROCESSING_TIME: {round(time.time() - start, 3)}]')
    logger.info(f'[TOTAL_DOCUMENTS_PROCESSED: {total_documents_processed}]')
    return {"data": return_data}
