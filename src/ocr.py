import os
import json
import sys
import io

# Reconfigure standard output and error to use UTF-8.
# This prevents UnicodeEncodeError when EasyOCR prints its progress bar (\u2588) to a Windows console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import easyocr

class OCRExtractor:
    def __init__(self, cache_dir="outputs/ocr_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        # Initialize the EasyOCR reader
        print("[OCR] Initializing EasyOCR Reader on CPU...")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image_path):
        filename = os.path.basename(image_path)
        cache_path = os.path.join(self.cache_dir, f"{filename}.json")

        # Check if cached version exists
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # Return the list of text dicts
                    return cached_data.get("results", [])
            except Exception as e:
                print(f"[OCR] Error reading cache for {filename}: {e}. Re-running OCR...")

        # Run OCR
        try:
            raw_results = self.reader.readtext(image_path)
        except Exception as e:
            print(f"[OCR] Failed to run OCR on {filename}: {e}")
            return []

        # Convert to JSON-serializable types (numpy floats/ints to native Python types)
        serializable_results = []
        for box, text, confidence in raw_results:
            clean_box = [[float(coord) for coord in pt] for pt in box]
            serializable_results.append({
                "box": clean_box,
                "text": str(text),
                "confidence": float(confidence)
            })

        # Save to cache
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "image_name": filename,
                    "results": serializable_results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[OCR] Error saving cache for {filename}: {e}")

        return serializable_results

    def extract_plain_text(self, image_path):
        """Helper to get text lines only as a list of strings."""
        results = self.extract_text(image_path)
        return [item["text"] for item in results]
