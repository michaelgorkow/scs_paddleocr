import logging
import sys
from fastapi import FastAPI, Request
from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F
import requests
import os
import time
import json
import fitz
from PIL import Image
import cv2
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import base64
from io import BytesIO
import numpy as np

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
logger = get_logger('logger-visualize-extraction-webservice')

# To get better resolution when turning PDF pages to images for OCR
ZOOM_X = float(os.getenv('ZOOM_X'))                             # horizontal zoom
ZOOM_Y = float(os.getenv('ZOOM_Y'))                             # vertical zoom
mat = fitz.Matrix(ZOOM_X, ZOOM_Y)                               # zoom factor for each dimension

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

def get_connection_params():
    # Environment variables below will be automatically populated by Snowflake.
    snowflake_connection_cfg = {
       "HOST": os.getenv('SNOWFLAKE_HOST'),
       "ACCOUNT": os.getenv('SNOWFLAKE_ACCOUNT'),
       "TOKEN": get_login_token(),
       "authenticator": 'oauth',
       "warehouse": os.getenv('SNOWFLAKE_WAREHOUSE'),
       "database": os.getenv('SNOWFLAKE_DATABASE'),
       "schema": os.getenv('SNOWFLAKE_SCHEMA')
       }
    return snowflake_connection_cfg

# function to convert ocr bounding boxes
def ocr_bbox_to_x1y1x2y2(bbox):
    x1 = bbox[0][0]
    y1 = bbox[0][1]
    x2 = bbox[2][0]
    y2 = bbox[2][1]
    return [x1,y1,x2,y2]

@app.post("/visualize-extractions", tags=["Endpoints"])
async def extract_text(request: Request):
    # table: table with extraction outputs
    # stage_name: stage to files
    # doc_relative_path: relative path of document
    # doc_page_number: page number of document
    request_body = await request.json()
    request_body = request_body['data']
    return_data = []
    # create a snowflake session to retrieve file-urls while calling the extraction (avoid 1 hour limit of PRESIGNED_URLs)
    for index, table, stage_name, doc_relative_path, doc_page_number in request_body:
        session = Session.builder.configs(get_connection_params()).create()
        # Print out current session context information.
        database = session.get_current_database()
        schema = session.get_current_schema()
        warehouse = session.get_current_warehouse()
        role = session.get_current_role()
        logger.info(f"Connection succeeded. Current session context: database={database}, schema={schema}, warehouse={warehouse}, role={role}")
        # Retrieve all extracts for that document
        doc = session.table(table)
        doc = doc.filter(F.col('RELATIVE_PATH') == doc_relative_path)
        doc = doc.filter(F.col('OCR_PAGE_NUMBER') == doc_page_number)
        doc_download_url = doc.limit(1).with_column('PRESIGNED_URL', F.call_builtin('GET_PRESIGNED_URL', F.lit(stage_name), F.col('RELATIVE_PATH'))).collect()[0]['PRESIGNED_URL']
        logger.info(f'Number of extracted text objects: {doc.count()}')

        # Download PDF
        doc_bytes = requests.get(doc_download_url).content
        pdfdoc = fitz.open(stream=BytesIO(doc_bytes), filetype="pdf")
        pix = pdfdoc.get_page_pixmap(doc_page_number-1,matrix=mat)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

        # Annotate document with text extracts
        for row in doc[['OCR_LINE_NUMBER','OCR_TEXT','OCR_BBOX','OCR_CONFIDENCE']].order_by('OCR_LINE_NUMBER').to_local_iterator():
            bbox = json.loads(row['OCR_BBOX'])
            bbox = ocr_bbox_to_x1y1x2y2(bbox)
            img = cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,0,255), 2)
                
        #buffered = BytesIO()
        #img.save(buffered, format="JPEG")
        #img_str = base64.b64encode(buffered.getvalue())
        _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        img_str = base64.b64encode(im_bytes)
        return_data.append([index, img_str])
    return {"data": return_data}