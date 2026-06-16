import os
import re
import json
import math
import cv2
import numpy as np
import pytesseract
import pypdfium2 as pdfium
import io

# Setup Tesseract path
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "lab_reports")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "src", "medical_knowledge.json")
OCR_CACHE_PATH = os.path.join(OUTPUTS_DIR, "ocr_cache.json")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Predefined seed keywords for known/common classes to ensure high accuracy & backward compatibility
SEEDS = {
    "cbc": [
        "complete blood count", "cbc", "hemoglobin", "haemoglobin", "hb", "rbc", "wbc", "platelet", 
        "platelets", "hct", "mcv", "mch", "mchc", "rdw", "neutrophils", "lymphocytes", "monocytes", 
        "eosinophils", "basophils", "total leucocyte count", "tlc", "differential leucocyte count", "dlc"
    ],
    "lft": [
        "liver function test", "lft", "bilirubin", "total bilirubin", "direct bilirubin", "indirect bilirubin", 
        "sgot", "ast", "sgpt", "alt", "alp", "alkaline phosphatase", "albumin", "globulin", "total protein", "ggt"
    ],
    "kidney_function_test": [
        "kidney function test", "kft", "rft", "renal function test", "creatinine", "serum creatinine", 
        "urea", "blood urea", "bun", "uric acid", "egfr", "kidney profile", "renal profile"
    ],
    "crp": [
        "crp", "c-reactive protein", "hs-crp", "c reactive protein", "high sensitivity crp"
    ],
    "urine": [
        "urine routine", "urine analysis", "urinalysis", "cue", "specific gravity", "ph", "protein", 
        "glucose", "ketones", "pus cells", "epithelial cells", "casts", "crystals", "nitrite", "bacteria", 
        "complete urine examination"
    ],
    "microbiology": [
        "culture report", "culture and sensitivity", "growth", "colony", "sensitive", "resistant", 
        "antibiotic", "organism", "organisms", "bacterial culture", "fungal culture", "urine culture"
    ],
    "haematology": [
        "haematology", "blood examination", "hematology", "esr", "erythrocyte sedimentation rate", 
        "pt", "inr", "aptt", "coagulation", "clotting time", "bleeding time"
    ],
    "lipid_profile": [
        "lipid profile", "cholesterol", "triglycerides", "hdl", "ldl", "vldl", "lipid", "total cholesterol"
    ],
    "thyroid_profile": [
        "thyroid profile", "thyroid", "t3", "t4", "tsh", "triiodothyronine", "thyroxine", "thyroid stimulating hormone"
    ],
    "electrolytes": [
        "electrolytes", "sodium", "potassium", "chloride", "bicarbonate", "serum electrolytes"
    ],
    "widel": [
        "widal", "widel", "typhi", "salmonella", "salmonella typhi", "slide agglutination", "widal test"
    ],
    "rbs": [
        "rbs", "random blood sugar", "blood sugar", "glucose", "random glucose"
    ],
    "blood_grounping_rh": [
        "blood grouping", "blood group", "rh factor", "abo", "coombs", "blood group rh"
    ],
    "vitamin_d": [
        "vitamin d", "25-hydroxyvitamin d", "cholecalciferol", "vit d", "vitamin d3"
    ],
    "vitamin_b12": [
        "vitamin b12", "cobalamin", "vit b12", "cyanocobalamin"
    ],
    "abg": [
        "abg", "arterial blood gas", "pco2", "po2", "hco3", "base excess", "ph"
    ],
    "coagulation": [
        "coagulation", "prothrombin time", "pt", "inr", "aptt", "fibrinogen"
    ]
}

# Stopwords & Generic noise words to filter
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

GENERIC_NOISE = {
    'hospital', 'report', 'laboratory', 'laboratories', 'patient', 'doctor', 'physician', 'referred', 'collected',
    'reported', 'received', 'date', 'time', 'age', 'sex', 'gender', 'result', 'units', 'range', 'normal', 'low',
    'high', 'flag', 'reference', 'interval', 'value', 'signature', 'end', 'printed', 'receptionist', 'authorized',
    'page', 'phone', 'mobile', 'tel', 'fax', 'email', 'address', 'street', 'road', 'mumbai', 'delhi', 'india',
    'pathologist', 'opinion', 'clinical', 'correlation', 'diagnostic', 'diagnostics', 'test', 'tests', 'profile',
    'panel', 'examination', 'sample', 'specimen', 'billing', 'invoice', 'receipt', 'charge', 'discharge', 'summary',
    'history', 'valid', 'purpose', 'purposes', 'medico', 'legal', 'certified', 'end_of_report', 'department',
    'stationery', 'director', 'consultant', 'mbbs', 'md', 'dcp', 'phd', 'registration', 'center', 'centre',
    'health', 'care', 'medical', 'diagnose', 'diagnosis', 'report_id', 'number', 'dr', 'mrs', 'mr', 'miss',
    'hosp', 'lab', 'labs', 'inst', 'institute', 'telno', 'mob', 'emailid', 'website', 'web'
}

ALL_STOPWORDS = STOPWORDS.union(GENERIC_NOISE)

APPROVED_SHORT_TOKENS = {
    'hb', 'ph', 'pt', 'ca', 'mg', 'k', 'na', 'cl', 'crp', 'esr', 'lft', 'cbc', 'rft', 'kft', 'ast', 'alt', 'alp', 'bun',
    't3', 't4', 'tsh', 'rbs', 'fbs', 'abo', 'wbc', 'rbc', 'hct', 'mcv', 'mch', 'plt', 'pco2', 'po2', 'hco3', 'inr'
}

def clean_ocr_text(text: str) -> str:
    """
    Cleans raw OCR text and removes noise like patient info, doctor names, dates, phone numbers, and addresses.
    """
    if not text:
        return ""
    text = text.lower()
    
    # Remove emails and web links
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\b(https?://)?(www\.)?[\w.-]+\.\w+\b', ' ', text)
    
    # Remove dates
    text = re.sub(r'\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b', ' ', text)
    text = re.sub(r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}\b', ' ', text)
    text = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b', ' ', text)
    
    # Remove IDs, MRNs, registration parameters, phone numbers
    text = re.sub(r'\b(mrn|id|s\.?no|reg|ref|uhid|ipd|opd|bill|phone|mobile|tel|fax|npi)\s*[-:]?\s*[a-z0-9]+\b', ' ', text)
    text = re.sub(r'\bdr\.\s+[a-z]+(\s+[a-z]+)?\b', ' ', text)
    text = re.sub(r'\+?\d[\d\s-]{8,15}', ' ', text)
    
    # Remove standalone numbers (results, ranges, reference numbers)
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Keep only alphabetical characters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_candidate_ngrams(cleaned_text: str):
    """
    Extracts valid unigrams, bigrams, and trigrams from cleaned text.
    """
    tokens = cleaned_text.split()
    candidates = []
    
    def is_valid_token(w):
        if w in ALL_STOPWORDS:
            return False
        if len(w) <= 2 and w not in APPROVED_SHORT_TOKENS:
            return False
        return True

    # 1. Unigrams
    for t in tokens:
        if is_valid_token(t):
            candidates.append(t)
            
    # 2. Bigrams
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        if is_valid_token(w1) and is_valid_token(w2):
            candidates.append(f"{w1} {w2}")
            
    # 3. Trigrams
    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        if is_valid_token(w1) and is_valid_token(w2) and is_valid_token(w3):
            candidates.append(f"{w1} {w2} {w3}")
            
    return candidates

def preprocess_image_file(img_path) -> np.ndarray:
    """
    Apply premium denoising and thresholding to report image.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
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

def pdf_to_image_bytes(pdf_path) -> bytes:
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        doc = pdfium.PdfDocument(pdf_bytes)
        page = doc[0]
        bitmap = page.render(scale=2)
        pil_image = bitmap.to_pil()
        img_io = io.BytesIO()
        pil_image.save(img_io, format='PNG')
        return img_io.getvalue()
    except Exception as e:
        print(f"Error rendering PDF {pdf_path}: {e}")
        return None

def ocr_document(file_path):
    """
    Extracts text from image or PDF using Tesseract OCR.
    """
    if file_path.lower().endswith('.pdf'):
        img_bytes = pdf_to_image_bytes(file_path)
        if img_bytes is None:
            return ""
        nparr = np.frombuffer(img_bytes, np.uint8)
        preprocessed = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    else:
        preprocessed = preprocess_image_file(file_path)
        
    if preprocessed is None:
        return ""
        
    try:
        text = pytesseract.image_to_string(preprocessed, config='--psm 3', timeout=25)
    except Exception as e:
        print(f"Warning: OCR timed out or failed for {file_path}: {e}")
        text = ""
    return text.strip()

def build_knowledge():
    # 1. Discover all categories based on folders in dataset directory
    print(f"Scanning dataset folders inside: {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory does not exist: {DATASET_DIR}")
        return

    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    categories = sorted(folders)
    print(f"Found {len(categories)} categories: {categories}")

    # 2. Initialize or load OCR Cache
    ocr_cache = {}
    if os.path.exists(OCR_CACHE_PATH):
        try:
            with open(OCR_CACHE_PATH, 'r', encoding='utf-8') as f:
                ocr_cache = json.load(f)
            print(f"Loaded OCR Cache from {OCR_CACHE_PATH} with {len(ocr_cache)} entries.")
        except Exception as e:
            print(f"Could not load OCR Cache: {e}. Starting fresh.")

    # 3. Perform OCR on all files
    documents_by_category = {}
    total_files = 0
    cache_hits = 0
    cache_misses = 0

    for cat in categories:
        documents_by_category[cat] = []
        cat_dir = os.path.join(DATASET_DIR, cat)
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf'))]
        
        for f in files:
            full_path = os.path.join(cat_dir, f)
            rel_path = os.path.relpath(full_path, BASE_DIR).replace('\\', '/')
            
            # File stats for cache validation
            stat = os.stat(full_path)
            size = stat.st_size
            mtime = stat.st_mtime
            
            # Check cache
            cached = ocr_cache.get(rel_path)
            if cached and cached.get('size') == size and cached.get('mtime') == mtime:
                ocr_text = cached.get('text', '')
                cache_hits += 1
            else:
                print(f"Running OCR on new file: {rel_path}...")
                ocr_text = ocr_document(full_path)
                ocr_cache[rel_path] = {
                    'text': ocr_text,
                    'size': size,
                    'mtime': mtime
                }
                cache_misses += 1
                try:
                    with open(OCR_CACHE_PATH, 'w', encoding='utf-8') as f_cache:
                        json.dump(ocr_cache, f_cache, indent=2)
                except Exception as cache_err:
                    print(f"Warning: could not save incremental cache: {cache_err}")
            
            documents_by_category[cat].append({
                'filename': f,
                'raw_text': ocr_text,
                'clean_text': clean_ocr_text(ocr_text)
            })
            total_files += 1

    print(f"OCR processing completed. Cache Hits: {cache_hits}, Cache Misses/OCR Runs: {cache_misses}")

    # Save updated OCR cache
    with open(OCR_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(ocr_cache, f, indent=2)
    print(f"OCR cache saved to {OCR_CACHE_PATH}")

    # 4. Extract Keywords using TF-IDF metric
    # Let's count how many documents in each category contain each candidate n-gram
    category_vocab = {}  # {cat: {term: doc_count}}
    all_vocab = set()
    
    for cat, docs in documents_by_category.items():
        category_vocab[cat] = {}
        for doc in docs:
            # Get unique candidates in this document to calculate document frequency
            candidates = set(extract_candidate_ngrams(doc['clean_text']))
            for cand in candidates:
                category_vocab[cat][cand] = category_vocab[cat].get(cand, 0) + 1
                all_vocab.add(cand)

    # Calculate IDF for all terms
    # IDF(t) = log((1 + Total Categories) / (1 + Categories containing t)) + 1
    num_categories = len(categories)
    idf = {}
    for term in all_vocab:
        cats_with_term = sum(1 for cat in categories if term in category_vocab[cat])
        idf[term] = math.log((1.0 + num_categories) / (1.0 + cats_with_term)) + 1.0

    # Calculate TF-IDF score for each term in each category and select top keywords
    knowledge_base = {}

    for cat in categories:
        docs = documents_by_category[cat]
        num_docs = len(docs)
        if num_docs == 0:
            knowledge_base[cat] = {"keywords": []}
            continue

        scores = {}
        for term, count in category_vocab[cat].items():
            # Only keep terms appearing in at least 2 docs (or 1 if cat has only 1 doc)
            min_docs_req = min(2, num_docs)
            if count < min_docs_req:
                continue
            
            tf = count / num_docs
            scores[term] = tf * idf[term]

        # Sort terms by score descending
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Select top 20 terms
        top_terms = [term for term, score in sorted_terms[:20]]

        # Merge with predefined seeds if present
        norm_cat = cat.lower().strip()
        seed_keywords = SEEDS.get(norm_cat, [])
        
        # Merge, deduplicate, and sort alphabetically
        final_keywords_set = set(top_terms).union(seed_keywords)
        # Always make sure the category name itself is in keywords (both raw and formatted)
        final_keywords_set.add(norm_cat)
        final_keywords_set.add(norm_cat.replace('_', ' '))
        
        final_keywords = sorted(list(final_keywords_set))
        knowledge_base[cat] = {
            "keywords": final_keywords
        }
        print(f"Category '{cat}' Keywords count: {len(final_keywords)}")

    # 5. Write to medical_knowledge.json
    with open(KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2)
    print(f"Successfully compiled medical knowledge base and saved to {KNOWLEDGE_PATH}")

if __name__ == "__main__":
    build_knowledge()
