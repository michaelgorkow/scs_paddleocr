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
   conn = connect(
      host = os.getenv('SNOWFLAKE_HOST'),
      account = os.getenv('SNOWFLAKE_ACCOUNT'),
      token = get_login_token(),
      authenticator = 'oauth'
   )
   start = time.time()
   for index, stage_name, relative_path in request_body:
      logger.info(f'DOCUMENT_PATH:{relative_path}')
      relative_path = escape_path(relative_path)
      sql_string = f"SELECT GET_PRESIGNED_URL('{stage_name}','{relative_path}')"
      result = conn.cursor().execute(sql_string).fetchall()
      document_file = result[0][0]
      extraction_results = ''
      document_file = requests.get(document_file, verify=False).content
      _, file_extension = os.path.splitext(relative_path)
      file_extension = file_extension.lower()
      # pdf extraction
      if file_extension == '.pdf':
         extraction_results = extract_pdf(document_file, relative_path)
      total_documents_processed += 1
      return_data.append([index, extraction_results])
   logger.info(f'[TOTAL_DOCUMENTS_PROCESSED: {total_documents_processed}] [DOCUMENT: {relative_path}] [PROCESSING_TIME: {time.time() - start}]')
   return {"data": return_data}