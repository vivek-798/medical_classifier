import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytesseract
from transformers import AutoProcessor, LayoutLMv3ForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure Tesseract path
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH

# Set UTF-8 encoding wrapper for console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8') if hasattr(sys.stderr, 'buffer') else sys.stderr

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "lab_reports")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "layoutlmv3_classifier")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(MODEL_SAVE_DIR, "ocr_cache.pkl")

TARGET_CLASSES = ["cbc", "crp", "lft", "kidney_function_test", "urine", "microbiology", "haematology"]

def get_dataset_df():
    rows = []
    for cls in TARGET_CLASSES:
        folder = os.path.join(DATASET_PATH, cls)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                rows.append({
                    'image_path': os.path.join(folder, f),
                    'image_name': f,
                    'label': cls
                })
    df = pd.DataFrame(rows)
    return df

def pre_cache_ocr(df):
    if os.path.exists(CACHE_PATH):
        print(f"Loading pre-cached OCR data from {CACHE_PATH}...")
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)
            
    print("Pre-extracting OCR data for all images using Tesseract...")
    cache = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="OCR Pre-extraction"):
        img_path = row['image_path']
        try:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            words = []
            boxes = []
            for i in range(len(ocr_data['text'])):
                word = ocr_data['text'][i].strip()
                if word != "":
                    words.append(word)
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    
                    x0 = max(0, min(1000, int(x / width * 1000)))
                    y0 = max(0, min(1000, int(y / height * 1000)))
                    x1 = max(0, min(1000, int((x + w) / width * 1000)))
                    y1 = max(0, min(1000, int((y + h) / height * 1000)))
                    
                    boxes.append([x0, y0, x1, y1])
            if not words:
                words = ["missing"]
                boxes = [[0, 0, 0, 0]]
            cache[img_path] = {
                'words': words,
                'boxes': boxes
            }
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            cache[img_path] = {
                'words': ["missing"],
                'boxes': [[0, 0, 0, 0]]
            }
            
    print(f"Saving pre-cached OCR data to {CACHE_PATH}...")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)
    return cache

class LayoutLMv3MedicalDataset(Dataset):
    def __init__(self, df, processor, ocr_cache, max_len=512):
        self.df = df
        self.processor = processor
        self.ocr_cache = ocr_cache
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = row['label_encoded']
        
        if image_path in self.ocr_cache:
            words = self.ocr_cache[image_path]['words']
            boxes = self.ocr_cache[image_path]['boxes']
        else:
            words = ["missing"]
            boxes = [[0, 0, 0, 0]]
            
        img = Image.open(image_path).convert("RGB")
        
        encoding = self.processor(
            img,
            text=words,
            boxes=boxes,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bbox': encoding['bbox'].squeeze(0),
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    df = get_dataset_df()
    print(f"Total report images found: {df.shape[0]}")
    
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    label_encoder_path = os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to: {label_encoder_path}")
    
    ocr_cache = pre_cache_ocr(df)
    
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['label_encoded']
    )
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    processor.save_pretrained(MODEL_SAVE_DIR)
    
    print("Loading model...")
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(TARGET_CLASSES)
    )
    
    # Freeze lower layers
    for param in model.layoutlmv3.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    model = model.to(device)
    
    # Feature Extraction (caching output CLS tokens of frozen backbone)
    print("\nPre-extracting LayoutLMv3 CLS features to speed up CPU training...")
    
    def extract_features(df_subset, desc):
        dataset = LayoutLMv3MedicalDataset(df_subset, processor, ocr_cache)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        all_cls_features = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bbox = batch['bbox'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels']
                
                outputs = model.layoutlmv3(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values
                )
                cls_feat = outputs[0][:, 0, :].cpu()
                all_cls_features.append(cls_feat)
                all_labels.append(labels)
                
        features_tensor = torch.cat(all_cls_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        return features_tensor, labels_tensor

    feature_start = time.time()
    train_features, train_labels = extract_features(train_df, "Train Feature Extraction")
    val_features, val_labels = extract_features(val_df, "Val Feature Extraction")
    print(f"Feature extraction completed in {time.time() - feature_start:.1f}s")
    
    train_loader = DataLoader(FeatureDataset(train_features, train_labels), batch_size=8, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_features, val_labels), batch_size=8)
    
    # Calculate class weights to handle imbalance
    class_counts = df['label_encoded'].value_counts().sort_index().values
    total_samples = len(df)
    num_classes = len(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.classifier.parameters(), lr=2e-4, weight_decay=0.05)
    
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    epochs = 40
    
    print("\n--- Starting Training ---")
    training_start = time.time()
    epoch_times = []
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.unsqueeze(1).to(device) # shape (batch_size, 1, hidden_size)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model.classifier(batch_features).squeeze(1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += len(batch_labels)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        model.classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.unsqueeze(1).to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model.classifier(batch_features).squeeze(1)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item() * len(batch_labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += len(batch_labels)
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch:02d}/{epochs}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.3f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(MODEL_SAVE_DIR)
            print("=> Saved best model checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break
                
    total_training_time = time.time() - training_start
    avg_epoch_time = np.mean(epoch_times)
    print(f"\nTraining completed in {total_training_time:.1f}s (Avg per epoch: {avg_epoch_time:.3f}s)")
    
    # Final Evaluation
    print("\n--- Running Final Evaluation on Validation Set ---")
    best_model = LayoutLMv3ForSequenceClassification.from_pretrained(MODEL_SAVE_DIR).to(device)
    best_model.eval()
    
    all_preds = []
    all_labels = []
    image_names = []
    
    with torch.no_grad():
        for idx in range(len(val_df)):
            row = val_df.iloc[idx]
            image_path = row['image_path']
            image_name = row['image_name']
            label = row['label_encoded']
            
            words = ocr_cache.get(image_path, {'words': ["missing"], 'boxes': [[0,0,0,0]]})['words']
            boxes = ocr_cache.get(image_path, {'words': ["missing"], 'boxes': [[0,0,0,0]]})['boxes']
            img = Image.open(image_path).convert("RGB")
            
            inputs = processor(
                img,
                text=words,
                boxes=boxes,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            bbox = inputs['bbox'].to(device)
            pixel_values = inputs['pixel_values'].to(device)
            
            outputs = best_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values
            )
            
            probs = torch.softmax(outputs.logits, dim=1).flatten()
            pred_idx = torch.argmax(probs).item()
            
            all_preds.append(pred_idx)
            all_labels.append(label)
            image_names.append(image_name)
            
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    print("\n=================== VALIDATION METRICS ===================")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("==========================================================")
    
    print("\nClassification Report:")
    target_names = label_encoder.inverse_transform(range(len(TARGET_CLASSES)))
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.title('LayoutLMv3 Document Classifier Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(MODEL_SAVE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to: {cm_path}")
    
    print("\nSample Validation Predictions:")
    pred_labels = label_encoder.inverse_transform(all_preds)
    actual_labels = label_encoder.inverse_transform(all_labels)
    
    pred_df = pd.DataFrame({
        'Image': image_names,
        'Actual': actual_labels,
        'Predicted': pred_labels,
        'Correct': [a == p for a, p in zip(actual_labels, pred_labels)]
    })
    print(pred_df.head(20).to_markdown(index=False))
    
    metrics_path = os.path.join(MODEL_SAVE_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        import json
        json.dump({
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_epoch_time': float(avg_epoch_time),
            'total_training_time': float(total_training_time)
        }, f, indent=2)
    print(f"Validation metrics saved to: {metrics_path}")

if __name__ == "__main__":
    train()
