import os
import json
import numpy as np
from medical_classifier import MedicalKeywordClassifier
import build_medical_knowledge  # import for cleaning & ocr fallback

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "lab_reports")
OCR_CACHE_PATH = os.path.join(BASE_DIR, "outputs", "ocr_cache.json")

def evaluate_classifier():
    print("Initializing upgraded MedicalKeywordClassifier...")
    classifier = MedicalKeywordClassifier()
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return
        
    # Load OCR cache
    ocr_cache = {}
    if os.path.exists(OCR_CACHE_PATH):
        try:
            with open(OCR_CACHE_PATH, 'r', encoding='utf-8') as f:
                ocr_cache = json.load(f)
            print(f"Loaded OCR Cache with {len(ocr_cache)} entries.")
        except Exception as e:
            print(f"Warning: Could not load OCR cache: {e}")
            
    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    categories = sorted(folders)
    
    actual_labels = []
    predicted_labels = []
    results = []
    misclassified = []
    
    total_scanned = 0
    
    for actual_cat in categories:
        cat_dir = os.path.join(DATASET_DIR, actual_cat)
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf'))]
        
        for f in files:
            full_path = os.path.join(cat_dir, f)
            rel_path = os.path.relpath(full_path, BASE_DIR).replace('\\', '/')
            
            # Retrieve text from cache or do OCR
            cached = ocr_cache.get(rel_path)
            if cached and cached.get('text'):
                text = cached['text']
            else:
                # Fallback to OCR if missing from cache
                text = build_medical_knowledge.ocr_document(full_path)
                
            # Run classifier
            pred_res = classifier.classify(text)
            pred_cat = pred_res["category"]
            confidence = pred_res["confidence"]
            matched = pred_res["matched_keywords"]
            
            # Normalize for comparison
            actual_norm = actual_cat.lower().strip()
            pred_norm = pred_cat.lower().strip()
            
            actual_labels.append(actual_norm)
            predicted_labels.append(pred_norm)
            total_scanned += 1
            
            is_correct = (actual_norm == pred_norm)
            
            res_item = {
                "filename": f,
                "rel_path": rel_path,
                "actual": actual_norm,
                "predicted": pred_norm,
                "confidence": confidence,
                "matched_keywords": matched,
                "correct": is_correct
            }
            results.append(res_item)
            
            if not is_correct:
                misclassified.append(res_item)

    print(f"\nEvaluation complete. Scanned {total_scanned} documents.")
    
    # 1. Calculate metrics per class
    unique_classes = sorted(list(set(actual_labels).union(set(predicted_labels))))
    
    class_metrics = {}
    total_correct = 0
    
    for cls in unique_classes:
        tp = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == cls and p == cls)
        fp = sum(1 for a, p in zip(actual_labels, predicted_labels) if a != cls and p == cls)
        fn = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == cls and p != cls)
        support = sum(1 for a in actual_labels if a == cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        total_correct += tp

    # Calculate overall metrics
    accuracy = total_correct / total_scanned if total_scanned > 0 else 0.0
    
    macro_precision = np.mean([m["precision"] for m in class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in class_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in class_metrics.values()])
    
    # Weighted metrics
    total_support = sum(m["support"] for m in class_metrics.values())
    if total_support > 0:
        weighted_precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / total_support
        weighted_recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_support
        weighted_f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    print("=" * 60)
    print("CLASSIFICATION EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_scanned} correct)")
    print(f"Macro Precision:  {macro_precision * 100:.2f}%")
    print(f"Macro Recall:     {macro_recall * 100:.2f}%")
    print(f"Macro F1 Score:   {macro_f1 * 100:.2f}%")
    print("-" * 60)
    print(f"Weighted Precision: {weighted_precision * 100:.2f}%")
    print(f"Weighted Recall:    {weighted_recall * 100:.2f}%")
    print(f"Weighted F1 Score:  {weighted_f1 * 100:.2f}%")
    print("=" * 60)

    # Class-wise detailed table
    print(f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 70)
    for cls in sorted(class_metrics.keys()):
        m = class_metrics[cls]
        if m["support"] > 0:  # Only show classes present in actual data
            print(f"{cls:<25} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1']*100:>8.2f}% | {m['support']:<8}")
    print("=" * 60)

    # Text-based Confusion Matrix
    print("\nCONFUSION MATRIX")
    print("=" * 60)
    # Let's show classes with actual support > 0
    matrix_classes = [c for c in unique_classes if class_metrics.get(c, {}).get("support", 0) > 0]
    
    # Header
    print(f"{'Actual \\ Predicted':<25} | " + " ".join(f"{c[:5]:<5}" for c in matrix_classes))
    print("-" * (28 + 6 * len(matrix_classes)))
    
    for act_cls in matrix_classes:
        row = []
        for pred_cls in matrix_classes:
            count = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == act_cls and p == pred_cls)
            row.append(f"{count:<5}")
        print(f"{act_cls:<25} | " + " ".join(row))
    print("=" * 60)

    # List of misclassifications
    print(f"\nMISCLASSIFIED DOCUMENTS ({len(misclassified)} total):")
    print("=" * 60)
    for m in misclassified[:30]:  # Limit output to top 30
        print(f"File: {m['filename']}")
        print(f"  Path: {m['rel_path']}")
        print(f"  Actual Label:    {m['actual']}")
        print(f"  Predicted Label: {m['predicted']} (Confidence: {m['confidence']})")
        print(f"  Matched Keys:    {m['matched_keywords']}")
        print("-" * 40)
    if len(misclassified) > 30:
        print(f"... and {len(misclassified) - 30} more misclassifications.")

if __name__ == "__main__":
    evaluate_classifier()
