# PaddleOCR in Snowpark Container Services
This repository explains how to run [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md) in Snowpark Container Services.  
Afterwards you can easily extract text from PDF documents via state-of-the-art Optical Character Recognition (OCR)  
## Requirements
* Account with Snowpark Container Services

## Setup Instructions
### 1. Setup required database objects
```sql
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;

-- Database Setup
CREATE DATABASE IF NOT EXISTS OCR_DEMO;
USE DATABASE OCR_DEMO;
CREATE STAGE IF NOT EXISTS DOCUMENTS ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = (ENABLE = TRUE);
CREATE STAGE IF NOT EXISTS CONTAINER_FILES ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = (ENABLE = TRUE);

-- Image Repository Setup
CREATE IMAGE REPOSITORY OCR_DEMO.PUBLIC.IMAGE_REPOSITORY;

-- Compute Pools Setup
CREATE COMPUTE POOL IF NOT EXISTS PADDLE_OCR_GPU_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = GPU_NV_S;

CREATE COMPUTE POOL IF NOT EXISTS VIZ_CPU_POOL
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = CPU_X64_XS;

-- Creating External Access Integration to allow web access (e.g. for downloading model files)
CREATE OR REPLACE NETWORK RULE CONTAINER_NETWORK_RULE
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = ('0.0.0.0:443','0.0.0.0:80');
    
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CONTAINER_ACCESS_INTEGRATION
    ALLOWED_NETWORK_RULES = (CONTAINER_NETWORK_RULE)
    ENABLED = true;

-- Create a new role for this workload and grant privileges
CREATE OR REPLACE ROLE CONTAINER_ROLE;
GRANT ALL ON DATABASE OCR_DEMO TO ROLE CONTAINER_ROLE;
GRANT ALL ON SCHEMA OCR_DEMO.PUBLIC TO ROLE CONTAINER_ROLE;
GRANT ALL ON STAGE OCR_DEMO.PUBLIC.CONTAINER_FILES TO ROLE CONTAINER_ROLE;
GRANT ALL ON STAGE OCR_DEMO.PUBLIC.DOCUMENTS TO ROLE CONTAINER_ROLE;
GRANT ALL ON WAREHOUSE COMPUTE_WH TO ROLE CONTAINER_ROLE;
GRANT ALL ON COMPUTE POOL PADDLE_OCR_GPU_POOL TO ROLE CONTAINER_ROLE;
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE CONTAINER_ROLE;
GRANT ALL ON INTEGRATION CONTAINER_ACCESS_INTEGRATION TO ROLE CONTAINER_ROLE;
GRANT READ ON IMAGE REPOSITORY IMAGE_REPOSITORY TO ROLE CONTAINER_ROLE;
GRANT ROLE CONTAINER_ROLE TO USER "ADMIN";
```

### 2. Clone this repository
```bash
git clone https://github.com/michaelgorkow/scs_paddleocr.git
```

### 3. Build & Upload the container
Make sure to add your organization- and accountname.  
```cmd
cd scs_paddleocr
docker build --platform linux/amd64 -t <ORGNAME>-<ACCTNAME>.registry.snowflakecomputing.com/ocr_demo/public/image_repository/pymupdf_paddleocr:latest container-pymupdf-paddleocr
docker push <ORGNAME>-<ACCTNAME>.registry.snowflakecomputing.com/ocr_demo/public/image_repository/pymupdf_paddleocr:latest
docker build --platform linux/amd64 -t <ORGNAME>-<ACCTNAME>.registry.snowflakecomputing.com/ocr_demo/public/image_repository/visualize_extractions:latest container-visualize-extractions
docker push <ORGNAME>-<ACCTNAME>.registry.snowflakecomputing.com/ocr_demo/public/image_repository/visualize_extractions:latest
```

### 4. Upload files to stages  
Use your favourite way of uploading files and upload 
* the `ocr_spec.yml` to stage `CONTAINER_FILES`
* the `viz_spec.yml` to stage `CONTAINER_FILES`
* PDF files to stage `DOCUMENTS`

### 5. Create the PaddleOCR and Vizualisation Service
```sql
-- Create Services
CREATE SERVICE SERVICE_PYMUPDF_PADDLEOCR
    IN COMPUTE POOL PADDLE_OCR_GPU_POOL
    FROM @CONTAINER_FILES
    SPEC='ocr_spec.yml'
    MIN_INSTANCES=1
    MAX_INSTANCES=3
    EXTERNAL_ACCESS_INTEGRATIONS = (CONTAINER_ACCESS_INTEGRATION);

CREATE SERVICE SERVICE_VIZ_OCR
    IN COMPUTE POOL VIZ_CPU_POOL
    FROM @CONTAINER_FILES
    SPEC='viz_spec.yml'
    MIN_INSTANCES=1
    MAX_INSTANCES=1
    EXTERNAL_ACCESS_INTEGRATIONS = (CONTAINER_ACCESS_INTEGRATION);

-- Verify Services are running
SELECT SYSTEM$GET_SERVICE_STATUS('SERVICE_PYMUPDF_PADDLEOCR');
SELECT SYSTEM$GET_SERVICE_STATUS('SERVICE_VIZ_OCR');
```

### 6. Create the service functions for OCR and visualizations
```sql
-- Function to run Optical Character Recognition on (OCR) on documents
CREATE OR REPLACE FUNCTION PYMUPDF_PADDLEOCR_EXTRACT(STAGE_NAME TEXT, RELATIVE_PATH TEXT)
    RETURNS VARIANT
    SERVICE=SERVICE_PYMUPDF_PADDLEOCR
    ENDPOINT=API
    MAX_BATCH_ROWS=1
    AS '/pymupdf-paddleocr-extract';

CREATE OR REPLACE FUNCTION VISUALIZE_EXTRACTIONS(TABLE_NAME TEXT, STAGE_NAME TEXT, RELATIVE_PATH TEXT, PAGE_NUMBER INT)
    RETURNS STRING
    SERVICE=SERVICE_VIZ_OCR
    ENDPOINT=API
    MAX_BATCH_ROWS=1
    AS '/visualize-extractions';
```

### 7. Call the service functions using files from a Directory Table and finally generate an annotated image of a Page
```sql
-- Create a table with raw extractions
CREATE OR REPLACE TABLE RAW_EXTRACTS AS (
    SELECT RELATIVE_PATH, 
           PYMUPDF_PADDLEOCR_EXTRACT('@OCR_DEMO.PUBLIC.DOCUMENTS',RELATIVE_PATH) AS OCR_RESULTS
    FROM DIRECTORY('@OCR_DEMO.PUBLIC.DOCUMENTS')
);

-- Transform extractions into lines
-- 1 row = 1 extracted line from the documents
CREATE OR REPLACE TABLE LINE_LEVEL_EXTRACTS AS (
    SELECT relative_path,
           ocr_data.index::INTEGER AS OCR_PAGE_NUMBER,
           page_level_data.index::INTEGER AS OCR_LINE_NUMBER,
           page_level_data.value[0]::ARRAY AS OCR_BBOX,
           page_level_data.value[1][0]::STRING AS OCR_TEXT,
           page_level_data.value[1][1]::FLOAT AS OCR_CONFIDENCE,
           page_rotations.value::INT AS PAGE_ROTATION
    FROM RAW_EXTRACTS,
     LATERAL FLATTEN(input => OCR_RESULTS['OCR_RESULTS']) ocr_data,
     LATERAL FLATTEN(input => ocr_data.value) page_level_data,
     LATERAL FLATTEN(input => OCR_RESULTS['PAGE_ROTATIONS']) page_rotations
    HAVING OCR_PAGE_NUMBER = page_rotations.index
);

-- Transform lines into pages
-- 1 row = 1 page
CREATE OR REPLACE TABLE PAGE_LEVEL_EXTRACTS AS (
    SELECT RELATIVE_PATH, 
           OCR_PAGE_NUMBER,
           LISTAGG(OCR_TEXT, ' ') WITHIN GROUP (ORDER BY OCR_LINE_NUMBER ASC) AS OCR_PAGE_TEXT,
           AVG(PAGE_ROTATION) AS PAGE_ROTATION
    FROM LINE_LEVEL_EXTRACTS
    GROUP BY RELATIVE_PATH,OCR_PAGE_NUMBER
);

-- Generate a base64 image of a specific page of a document
SELECT VISUALIZE_EXTRACTIONS('LINE_LEVEL_EXTRACTS', '@DOCUMENTS', 'md_hrs_de_folio_bermc_de5344672511.pdf', 0);
```

### 8. Stop Compute Ressources
```sql
-- Clean Up
DROP SERVICE SERVICE_PYMUPDF_PADDLEOCR;
DROP COMPUTE POOL PADDLE_OCR_GPU_POOL;
```

### Debugging: View Logs
If you want to know what's happening inside the container, you can retrieve the logs at any time.
```sql
-- See logs of container
SELECT value AS log_line
FROM TABLE(
 SPLIT_TO_TABLE(SYSTEM$GET_SERVICE_LOGS('PYMUPDF_PADDLEOCR_EXTRACT', 0, 'container-pymupdf-paddleocr'), '\n')
  );

SELECT value AS log_line
FROM TABLE(
 SPLIT_TO_TABLE(SYSTEM$GET_SERVICE_LOGS('SERVICE_VIZ_OCR', 0, 'container-visualize-extractions'), '\n')
  );
```
### Bonus: Streamlit PDF Viewer
We can use the VISUALIZE_EXTRACTIONS() function in Streamlit in Snowflake to visualize the outputs of our OCR pipeline.  
Simply add the code in `streamlit.py` to your Streamlit App.  

## Demo Video
