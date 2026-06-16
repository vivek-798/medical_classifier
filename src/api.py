import os
import sys
import tempfile
import logging
import numpy as np
import cv2
import pypdfium2 as pdfium
import pytesseract
import io
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import shutil

# Import modules
from medical_classifier import MedicalKeywordClassifier
import database

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MedicalAPI")

# Tesseract path configuration for local Windows environment (Render uses path-resolved binary)
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH
    logger.info(f"Using local Windows Tesseract path: {TESS_PATH}")
else:
    logger.info("Tesseract binary path not overridden. Using default system PATH binary.")

# Limit internal Tesseract threading for speed optimization
os.environ["OMP_THREAD_LIMIT"] = "1"

app = FastAPI(title="Medical Lab Report OCR & Classification API")

# Add CORS Middleware to support cross-origin requests from Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In a production environment, restrict to Netlify domains if desired
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = MedicalKeywordClassifier()

class ClassificationResponse(BaseModel):
    report_id: int
    filename: str
    category: str
    confidence: float
    matched_keywords: list[str]
    text: str

class FeedbackRequest(BaseModel):
    report_id: int
    user_confirmation: str  # "correct" or "incorrect"
    final_correct_label: str

def preprocess_image(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file format")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1500:
        scale = 1800.0 / w
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    return thresh

def pdf_first_page_to_image_bytes(pdf_bytes: bytes) -> bytes:
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        page = doc[0]
        bitmap = page.render(scale=2)
        pil_image = bitmap.to_pil()
        img_io = io.BytesIO()
        pil_image.save(img_io, format='PNG')
        return img_io.getvalue()
    except Exception as e:
        raise ValueError(f"Failed to process PDF file: {e}")

@app.on_event("startup")
def startup_event():
    logger.info("Initializing database verification...")
    database.init_db()

@app.post("/classify", response_model=ClassificationResponse)
def classify_report(file: UploadFile = File(...)):
    filename = file.filename
    content_type = file.content_type or ""
    logger.info(f"Received classify request for file: {filename} (Content-Type: {content_type})")
    
    temp_file_path = None
    try:
        file_bytes = file.file.read()
        
        # 1. Create a temporary file to store uploaded bytes during processing (Stateless requirement)
        _, ext = os.path.splitext(filename)
        temp_fd, temp_file_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(temp_fd, "wb") as tmp:
            tmp.write(file_bytes)
        logger.info(f"Saved uploaded file to temporary path: {temp_file_path}")
        
        # 2. Check if file is PDF
        if filename.lower().endswith(".pdf") or "pdf" in content_type.lower():
            logger.info("Processing document as PDF...")
            img_bytes = pdf_first_page_to_image_bytes(file_bytes)
            base_name, _ = os.path.splitext(filename)
            saved_filename = f"{base_name}.png"
        else:
            img_bytes = file_bytes
            saved_filename = filename
            
        # Preprocess
        logger.info("Preprocessing image for OCR...")
        preprocessed = preprocess_image(img_bytes)
        
        # Run Tesseract OCR directly on preprocessed image text
        logger.info("Executing PyTesseract OCR extraction...")
        text = pytesseract.image_to_string(preprocessed, config='--psm 3')
        text_clean = text.strip()
        logger.info(f"OCR extracted {len(text_clean)} characters of text.")
        
        # Classify Text using Medical Keyword Classifier
        logger.info("Running medical keyword classifier...")
        result = classifier.classify(text_clean)
        category = result["category"]
        confidence = result["confidence"]
        matched_keywords = result["matched_keywords"]
        logger.info(f"Classification result: {category} (Confidence: {confidence})")
        
        # 3. Log entry to Supabase database
        report_id = database.save_report(
            filename=filename,
            ocr_text=text_clean,
            predicted_category=category,
            confidence=confidence,
            matched_keywords=matched_keywords
        )
        
        return ClassificationResponse(
            report_id=report_id,
            filename=filename,
            category=category,
            confidence=confidence,
            matched_keywords=matched_keywords,
            text=text_clean
        )
        
    except ValueError as val_err:
        logger.error(f"Value Error during classification: {val_err}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.error(f"Internal error during classification: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")
    finally:
        # 4. Delete the temporary file after OCR processing (Stateless requirement)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_err:
                logger.error(f"Failed to delete temporary file {temp_file_path}: {cleanup_err}")

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    logger.info(f"Received feedback request for report ID: {req.report_id}")
    try:
        success = database.update_feedback(
            report_id=req.report_id,
            user_confirmation=req.user_confirmation,
            final_correct_label=req.final_correct_label
        )
        if not success:
            logger.warning(f"Report ID {req.report_id} not found in database for feedback update.")
            raise HTTPException(status_code=404, detail="Report ID not found")
        logger.info(f"Successfully recorded feedback for report ID: {req.report_id}")
        return {"status": "success", "message": "Feedback submitted successfully"}
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Error updating feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Internal database error: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Medical Lab Report OCR & Classification API"}
