# Import python packages
import streamlit as st
import cv2
import numpy
import base64
from PIL import Image
from io import BytesIO
from snowflake.snowpark.context import get_active_session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T

st.title("PDF Extraction Viewer")

# Get the current credentials
session = get_active_session()

# Get a Doc
output_table = 'LINE_LEVEL_EXTRACTS'
stage_name = '@DOCUMENTS'
doc_table = session.table(output_table)
docs = doc_table.select('RELATIVE_PATH').distinct().limit(100)
doc_sampled_relative_path = st.selectbox('Select a document ...', docs)

pages = doc_table.filter(F.col('RELATIVE_PATH')==doc_sampled_relative_path)
pages = pages.group_by(['OCR_PAGE_NUMBER']).agg(F.count('OCR_LINE_NUMBER').as_('LINE_COUNT'))
pages = pages.order_by(F.col('OCR_PAGE_NUMBER').asc())
doc_sampled_page_number = st.selectbox('Select a page ...', pages)

# Get all lines
doc = session.table(output_table)
doc = doc.filter(F.col('RELATIVE_PATH') == doc_sampled_relative_path)
doc = doc.filter(F.col('OCR_PAGE_NUMBER') == doc_sampled_page_number)
st.dataframe(doc.order_by('OCR_LINE_NUMBER')[['OCR_LINE_NUMBER','OCR_TEXT','OCR_CONFIDENCE']])

# Retrieve base64 encoded image via SPCS service
if st.button('Visualize Page!'):
    viz = session.sql(f"SELECT VISUALIZE_EXTRACTIONS('{output_table}', '{stage_name}', '{doc_sampled_relative_path}', {doc_sampled_page_number}) AS VISUALIZED_EXTRACTIONS").collect()
    viz = viz[0]['VISUALIZED_EXTRACTIONS']
    viz = Image.open(BytesIO(base64.b64decode(viz)))
    
    st.image(viz)