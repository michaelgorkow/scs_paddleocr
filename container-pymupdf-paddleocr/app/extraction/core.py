import logging
import sys
import os, io
import fitz
import numpy as np
import requests
import json
from paddleocr import PaddleOCR
from PIL import Image
import time
import traceback

# Logging
logger = logging.getLogger('logger-pymupdf-paddleocr-core')
logger.setLevel(logging.DEBUG)

import paddle
gpu_available  = paddle.device.is_compiled_with_cuda()
logger.info(f"[GPU available: {gpu_available}]")

# ENVIRONMENT VARIABLES
PADDLEOCR_LANGUAGE = os.getenv('PADDLEOCR_LANGUAGE')            # Language for OCR
MAX_PAGES = int(os.getenv('MAX_PAGES'))                         # Number of pages to extract per document
# To get better resolution when turning PDF pages to images for OCR
ZOOM_X = float(os.getenv('ZOOM_X'))                             # horizontal zoom
ZOOM_Y = float(os.getenv('ZOOM_Y'))                             # vertical zoom
mat = fitz.Matrix(ZOOM_X, ZOOM_Y)                               # zoom factor for each dimension
DET_LIMIT_SIDE_LEN = int(os.getenv('DET_LIMIT_SIDE_LEN'))       # maximum image length (either side) for text detection algorithm, multiples of 32 supported
DET_DB_UNCLIP_RATIO = float(os.getenv('DET_DB_UNCLIP_RATIO'))   # in/decrease area of crops
OUTPUT_FORMAT = os.getenv('OUTPUT_FORMAT')                              # enable simple output (useful for large documents that return >10 MB response hitting the service function limits)
if OUTPUT_FORMAT == 'SIMPLE':
    SIMPLE_OUTPUT_THRESHOLD = float(os.getenv('SIMPLE_OUTPUT_THRESHOLD'))   # Threshold to keep detected words in results

logger.info(f'[OCR Language: {PADDLEOCR_LANGUAGE}]')

# Load paddleocr
ocr = PaddleOCR(
   use_angle_cls=False, 
   lang=PADDLEOCR_LANGUAGE,
   det_model_dir='/stage_pymupdf_paddleocr/paddle/det_model_dir/',
   rec_model_dir='/stage_pymupdf_paddleocr/paddle/rec_model_dir/',
   cls_model_dir='/stage_pymupdf_paddleocr/paddle/cls_model_dir/',
   det_limit_side_len = DET_LIMIT_SIDE_LEN,
   cls_thresh = 0.9,
   use_gpu=True,
   show_log = False,
   save_crop_res=False,
   det_db_unclip_ratio=DET_DB_UNCLIP_RATIO
   )
logger.info('[Finished loading PaddleOCR model.]')

ROTATIONS_TO_TRY = [90,180,270]

# Function to simplify output
def simplify_output(output):
    output = [" ".join([val[1][0] for val in output if val[1][1] > SIMPLE_OUTPUT_THRESHOLD])]
    return output

# Function to extract content from PDFs
def extract_pdf(file, relative_path):
    start = time.time()
    filestream = io.BytesIO(file)
    ocr_results = []
    page_rotations = []
    page_no = 1
    try:
        doc = fitz.open(stream=filestream, filetype="pdf")  # open document
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f'[CURRENT DOCUMENT: {relative_path}] [ERROR_MESSAGE: {error_trace}]')
        return {'OCR_RESULTS':[], 'PAGE_ROTATIONS':[], 'ERROR_MESSAGE':error_trace}
    if not doc.page_count:
        logger.error(f'[CURRENT DOCUMENT: {relative_path}] [ERROR_MESSAGE: EMPTY_DOCUMENT]')
        return {'OCR_RESULTS':[], 'PAGE_ROTATIONS':[], 'ERROR_MESSAGE':'EMPTY_DOCUMENT'}
    for page in doc.pages():
        if page_no <= MAX_PAGES:
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            np_image = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3)) # turn page image into numpy array
            ocr_start = time.time()
            best_rotation = 0
            ocr_results_page = ocr.ocr(np_image)[0] # run ocr
            if ocr_results_page is not None:
                ocr_results_page_avg_conf = np.mean([val[1][1] for val in ocr_results_page])+0.05 # small bonus assuming most pages are not rotated
                ocr_results_page_num_boxes = len(ocr_results_page)
            else:
                ocr_results_page_avg_conf = 0
                ocr_results_page_num_boxes = 0
            if ocr_results_page_avg_conf < 0.9:
                # try different rotations if low confidence
                for rotation in ROTATIONS_TO_TRY:
                    img = np.array(Image.fromarray(np_image).rotate(rotation, expand=True)) # rotate
                    _ocr_results_page =  ocr.ocr(img)[0] # run ocr
                    if _ocr_results_page is None:
                        continue
                    else:
                        _ocr_results_page_avg_conf = np.mean([val[1][1] for val in _ocr_results_page])
                        _ocr_results_page_num_boxes = len(_ocr_results_page)
                    if _ocr_results_page_avg_conf > ocr_results_page_avg_conf and (_ocr_results_page_num_boxes >= ocr_results_page_num_boxes):
                        ocr_results_page_avg_conf = _ocr_results_page_avg_conf
                        ocr_results_page_num_boxes = _ocr_results_page_num_boxes
                        ocr_results_page = _ocr_results_page
                        best_rotation = rotation
                if _ocr_results_page is None:
                    # assuming all rotations yielded no results
                    ocr_results_page = ''
            page_rotations.append(best_rotation)
            # simple output
            if OUTPUT_FORMAT == 'SIMPLE':
                ocr_results_page = simplify_output(ocr_results_page)
            ocr_results.append(ocr_results_page)
            logger.info(f'[CURRENT DOCUMENT: {relative_path}] [PAGE_NO: {page_no}] [PAGE_OCR_TIME: {round(time.time() - ocr_start, 3)}]')
            page_no += 1
        else:
            break
    logger.info(f'[CURRENT DOCUMENT: {relative_path}] [TOTAL_PAGES: {page_no-1}] [TOTAL_OCR_TIME: {round(time.time() - start, 3)}]')
    return {'OCR_RESULTS':ocr_results, 'PAGE_ROTATIONS':page_rotations, 'ERROR_MESSAGE':''}
