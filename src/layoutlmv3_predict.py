import os
import sys
import pickle
import argparse
import json
import torch
from PIL import Image
import pytesseract
from transformers import AutoProcessor, LayoutLMv3ForSequenceClassification

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "layoutlmv3_classifier")

# Configure Tesseract path
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

class LayoutLMv3Classifier:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        le_path = os.path.join(self.model_dir, "label_encoder.pkl")
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"Label encoder not found at {le_path}. Please run training first.")
            
        # Load artifacts
        print(f"Loading LayoutLMv3 model from {self.model_dir}...")
        self.processor = AutoProcessor.from_pretrained(self.model_dir, apply_ocr=False)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(self.model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        with open(le_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def get_ocr_data(self, image_path):
        """
        Runs Tesseract OCR to extract words and normalized bounding boxes.
        """
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        words = []
        boxes = []
        
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].strip()
            if word != "":
                words.append(word)
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                
                # Normalize coordinates to 0-1000 scale
                x0 = max(0, min(1000, int(x / width * 1000)))
                y0 = max(0, min(1000, int(y / height * 1000)))
                x1 = max(0, min(1000, int((x + w) / width * 1000)))
                y1 = max(0, min(1000, int((y + h) / height * 1000)))
                
                boxes.append([x0, y0, x1, y1])
                
        if not words:
            words = ["missing"]
            boxes = [[0, 0, 0, 0]]
            
        return words, boxes, img

    def predict(self, image_path: str) -> dict:
        """
        Runs document layout & text inference on the given report image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Extract OCR data
        words, boxes, img = self.get_ocr_data(image_path)
        
        # Process inputs
        inputs = self.processor(
            img,
            text=words,
            boxes=boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        bbox = inputs['bbox'].to(self.device)
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values
            )
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).flatten()
            
            pred_idx = torch.argmax(probabilities).item()
            confidence = probabilities[pred_idx].item()
            
        prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "text": " ".join(words)
        }

def main():
    parser = argparse.ArgumentParser(description="LayoutLMv3 Classifier Inference CLI")
    parser.add_argument("image_path", type=str, help="Path to report image file")
    args = parser.parse_args()
    
    try:
        classifier = LayoutLMv3Classifier()
        result = classifier.predict(args.image_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
