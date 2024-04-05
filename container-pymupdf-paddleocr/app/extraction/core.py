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
logger = get_logger('logger-pymupdf-paddleocr-core')

import paddle
gpu_available  = paddle.device.is_compiled_with_cuda()
logger.info(f"GPU available: {gpu_available}")

# ENVIRONMENT VARIABLES
PADDLEOCR_LANGUAGE = os.getenv('PADDLEOCR_LANGUAGE')            # Language for OCR
MAX_PAGES = int(os.getenv('MAX_PAGES'))                         # Number of pages to extract per document
# To get better resolution when turning PDF pages to images for OCR
ZOOM_X = float(os.getenv('ZOOM_X'))                             # horizontal zoom
ZOOM_Y = float(os.getenv('ZOOM_Y'))                             # vertical zoom
mat = fitz.Matrix(ZOOM_X, ZOOM_Y)                               # zoom factor for each dimension
DET_LIMIT_SIDE_LEN = int(os.getenv('DET_LIMIT_SIDE_LEN'))       # maximum image length (either side) for text detection algorithm, multiples of 32 supported
DET_DB_UNCLIP_RATIO = float(os.getenv('DET_DB_UNCLIP_RATIO'))   # in/decrease area of crops

logger.info(f'OCR Language: {PADDLEOCR_LANGUAGE}')

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
logger.info('Finished loading PaddleOCR model.')

ROTATIONS_TO_TRY = [90,180,270]

# Function to extract content from PDFs
def extract_pdf(file):
    start = time.time()
    filestream = io.BytesIO(file)
    ocr_results = []
    page_rotations = []
    page_no = 0
    try:
        doc = fitz.open(stream=filestream, filetype="pdf")  # open document
        for page in doc.pages():
            if page_no < MAX_PAGES:
                try:
                    pix = page.get_pixmap(matrix=mat)  # render page to an image
                    np_image = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3)) # turn page image into numpy array
                    try:
                        ocr_start = time.time()
                        best_rotation = 0
                        ocr_results_page = ocr.ocr(np_image)[0] # run ocr
                        try:
                            ocr_results_page_avg_conf = np.mean([val[1][1] for val in ocr_results_page])+0.05 # small bonus assuming most pages are not rotated
                            ocr_results_page_num_boxes = len(ocr_results_page)
                        except:
                            ocr_results_page_avg_conf = 0
                            ocr_results_page_num_boxes = 0
                        if ocr_results_page_avg_conf < 0.9:
                            # try different rotations if low confidence
                            for rotation in ROTATIONS_TO_TRY:
                                img = np.array(Image.fromarray(np_image).rotate(rotation, expand=True)) # rotate
                                _ocr_results_page =  ocr.ocr(img)[0] # run ocr
                                try:
                                    _ocr_results_page_avg_conf = np.mean([val[1][1] for val in _ocr_results_page])
                                    _ocr_results_page_num_boxes = len(_ocr_results_page)
                                except Exception as e:
                                    logger.error(e)
                                    _ocr_results_page_avg_conf = 0
                                    _ocr_results_page_num_boxes = 0
                                if _ocr_results_page_avg_conf > ocr_results_page_avg_conf and (_ocr_results_page_num_boxes >= ocr_results_page_num_boxes):
                                    ocr_results_page_avg_conf = _ocr_results_page_avg_conf
                                    ocr_results_page_num_boxes = _ocr_results_page_num_boxes
                                    ocr_results_page = _ocr_results_page
                                    best_rotation = rotation
                        page_rotations.append(best_rotation)
                        ocr_results.append(ocr_results_page)
                        logger.info(f'OCR_TIME: {time.time() - ocr_start}')
                    except Exception as e:
                        logger.error(e)
                except Exception as e:
                    logger.error(e)
                page_no += 1
            else:
                break
    except Exception as e:
        logger.error(e)
    logger.info(f'EXTRACTPDF_TIME: {time.time() - start}')
    return {'OCR_RESULTS':ocr_results, 'PAGE_ROTATIONS':page_rotations}