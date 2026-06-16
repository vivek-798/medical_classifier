import os
import sys
import pickle
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set standard encoding to UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8') if hasattr(sys.stderr, 'buffer') else sys.stderr

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "clinicalbert_classifier")

class ClinicalBERTClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        tokenizer_path = self.model_dir
        model_path = self.model_dir
        le_path = os.path.join(self.model_dir, "label_encoder.pkl")
        
        # Validate existence
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"Label encoder not found at {le_path}. Please train the model first.")
            
        # Load artifacts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        with open(le_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text: str) -> dict:
        """
        Infers the class label and confidence score for the given raw text.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).flatten()
            
            # Find best class prediction
            pred_idx = torch.argmax(probabilities).item()
            confidence = probabilities[pred_idx].item()
            
        prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }

def main():
    parser = argparse.ArgumentParser(description="ClinicalBERT Classifier Inference CLI")
    parser.add_argument("text", type=str, help="Raw medical text to classify")
    args = parser.parse_args()
    
    try:
        classifier = ClinicalBERTClassifier()
        result = classifier.predict(args.text)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
