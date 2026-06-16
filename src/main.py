import os
import shutil
import csv
import argparse
import sys
from ocr import OCRExtractor
from classifier import ReportClassifier

# Set standard encoding to UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
INPUT_DIR = os.path.join(DATASETS_DIR, "lbmaske")
OUTPUT_DIR = os.path.join(DATASETS_DIR, "lab_reports")
OUTPUTS_LOG_DIR = os.path.join(BASE_DIR, "outputs")
CSV_PATH = os.path.join(OUTPUTS_LOG_DIR, "classification_results.csv")
LOG_PATH = os.path.join(OUTPUTS_LOG_DIR, "classification_log.txt")

def resolve_folder_path(category_name, base_output_dir):
    """
    Normalizes category_name and checks for matching folders under base_output_dir,
    reusing folders with typos (like 'widel', 'blood_grounping_rh', 'epi') if present.
    """
    cat_norm = category_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    # Pre-defined mappings for existing folders (including typos in the original folders)
    mappings = {
        "cbc": "cbc",
        "crp": "crp",
        "lft": "lft",
        "liver_profile": "lft",
        "liver_function_test": "lft",
        "kft": "kft",
        "kidney_profile": "kft",
        "kidney_function_test": "kft",
        "rft": "kft",
        "renal_function_test": "kft",
        "rbs": "rbs",
        "haematology": "haematology",
        "hematology": "haematology",
        "serology": "serology",
        "microbiology": "microbiology",
        "coagulation": "coagulation",
        "electrolytes": "electrolytes",
        
        "widal_test": "widel",
        "widal": "widel",
        "widel": "widel",
        
        "complete_urine_examination": "cue",
        "cue": "cue",
        
        "examination_of_peripheral_smear": "epi",
        "epe": "epi",
        "epi": "epi",
        
        "blood_grouping_and_rh": "blood_grounping_rh",
        "blood_grouping_&_rh": "blood_grounping_rh",
        "blood_grouping_rh": "blood_grounping_rh",
        
        "abg": "abg",
        "arterial_blood_gas": "abg",
        "blood_gas": "abg",
        
        "urine_analysis": "urine",
        "urine": "urine",
        
        "bacteriology": "microbiology",
        "blood_esr": "blood_esr",
        "unknown": "unknown"
    }
    
    mapped_name = mappings.get(cat_norm, cat_norm)
    
    # Check folder availability in sequence
    check_dirs = [mapped_name, cat_norm]
    if cat_norm in ["kft", "kidney_profile", "kidney_function_test", "rft", "renal_function_test"]:
        check_dirs.extend(["kft", "kidney_function_test", "rft"])
    if cat_norm in ["abg", "arterial_blood_gas", "blood_gas"]:
        check_dirs.extend(["abg", "blood_gas"])
    if cat_norm in ["urine_analysis", "urine", "complete_urine_examination", "cue"]:
        check_dirs.extend(["urine", "cue"])
        
    for d in check_dirs:
        full_p = os.path.join(base_output_dir, d)
        if os.path.isdir(full_p):
            return d, full_p
            
    # Default to using mapped_name for new folder creation
    target_dir_name = mapped_name
    full_p = os.path.join(base_output_dir, target_dir_name)
    return target_dir_name, full_p

def log_message(msg, log_file=None):
    print(msg)
    if log_file:
        try:
            log_file.write(msg + "\n")
            log_file.flush()
        except Exception as e:
            print(f"Error writing to log file: {e}")

def restore_files():
    """
    Scans all folders under output directory and moves all images back to input directory.
    This facilitates easy reproducibility and clean re-runs.
    """
    print(f"Restoring files from {OUTPUT_DIR} back to {INPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR):
        print("Output directory does not exist. Nothing to restore.")
        return
        
    os.makedirs(INPUT_DIR, exist_ok=True)
    count = 0
    
    # Traverse subdirectories of OUTPUT_DIR
    for root, dirs, files in os.walk(OUTPUT_DIR):
        # Skip unknown folder or process it too (we restore everything)
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, filename)
                dest_path = os.path.join(INPUT_DIR, filename)
                
                # Handle filename collisions in source
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(os.path.join(INPUT_DIR, f"{name}_{counter}{ext}")):
                        counter += 1
                    dest_path = os.path.join(INPUT_DIR, f"{name}_{counter}{ext}")
                
                try:
                    shutil.move(src_path, dest_path)
                    count += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
                    
    print(f"Restored {count} files back to {INPUT_DIR}.")

def main():
    parser = argparse.ArgumentParser(description="Medical Lab Report Classifier")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of images to process")
    parser.add_argument("--dry-run", action="store_true", help="Run classifier without moving files or writing folders")
    parser.add_argument("--restore", action="store_true", help="Restore all images from lab_reports back to labmaske")
    args = parser.parse_args()

    if args.restore:
        restore_files()
        return

    # Ensure directories exist
    os.makedirs(OUTPUTS_LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        print(f"Input directory does not exist: {INPUT_DIR}")
        return

    # Open log file
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    log_message("=== Medical Lab Report Classifier Initialized ===", log_file)
    log_message(f"Input Directory: {INPUT_DIR}", log_file)
    log_message(f"Output Directory: {OUTPUT_DIR}", log_file)
    if args.dry_run:
        log_message("!!! RUNNING IN DRY-RUN MODE (No files will be moved) !!!", log_file)

    # Initialize modules
    extractor = OCRExtractor()
    classifier = ReportClassifier()

    # Get list of images
    all_files = os.listdir(INPUT_DIR)
    images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)
    
    log_message(f"Found {total_images} images in input directory.", log_file)
    if args.limit:
        images = images[:args.limit]
        log_message(f"Limit set: Processing first {len(images)} images.", log_file)

    # Trackers for summary
    processed_count = 0
    category_counts = {}
    newly_created_folders = set()
    unknown_count = 0
    csv_rows = []

    # Process images
    for idx, image_name in enumerate(images, 1):
        image_path = os.path.join(INPUT_DIR, image_name)
        log_message(f"\n[{idx}/{len(images)}] Processing {image_name}...", log_file)
        
        # 1. OCR text extraction (uses cache if available)
        text_lines = extractor.extract_plain_text(image_path)
        log_message(f"  OCR extracted {len(text_lines)} text lines.", log_file)
        
        # 2. Classification
        category, confidence = classifier.classify(text_lines)
        log_message(f"  Classified as: {category} (Confidence: {confidence:.2f})", log_file)

        # 3. Resolve destination folder
        folder_name, folder_path = resolve_folder_path(category, OUTPUT_DIR)
        
        # Check if folder will be newly created
        folder_existed = os.path.isdir(folder_path)
        
        # Track statistics
        processed_count += 1
        if category == "Unknown":
            unknown_count += 1
        category_counts[category] = category_counts.get(category, 0) + 1

        csv_rows.append({
            "image_name": image_name,
            "predicted_category": category,
            "confidence_score": f"{confidence:.2f}" if confidence > 0 else "N/A"
        })

        # 4. Perform File Move (unless dry-run)
        if not args.dry_run:
            if not folder_existed:
                os.makedirs(folder_path, exist_ok=True)
                newly_created_folders.add(folder_name)
                log_message(f"  Created new directory: {folder_name}", log_file)

            dest_path = os.path.join(folder_path, image_name)
            
            # Handle filename collisions in target
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(image_name)
                counter = 1
                while os.path.exists(os.path.join(folder_path, f"{name}_{counter}{ext}")):
                    counter += 1
                dest_path = os.path.join(folder_path, f"{name}_{counter}{ext}")
                log_message(f"  Collision detected. Moving to: {os.path.basename(dest_path)}", log_file)

            try:
                shutil.move(image_path, dest_path)
                log_message(f"  Moved to: {folder_name}/", log_file)
            except Exception as e:
                log_message(f"  Error moving file: {e}", log_file)
        else:
            if not folder_existed:
                newly_created_folders.add(folder_name)
            log_message(f"  [Dry-run] Would move to: {folder_name}/", log_file)

    # 5. Write CSV file
    if not args.dry_run:
        try:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_name", "predicted_category", "confidence_score"])
                writer.writeheader()
                writer.writerows(csv_rows)
            log_message(f"\nSaved classification results to CSV: {CSV_PATH}", log_file)
        except Exception as e:
            log_message(f"\nError writing CSV: {e}", log_file)
    else:
        log_message(f"\n[Dry-run] Would save classification results to CSV: {CSV_PATH}", log_file)

    # 6. Display final summary
    summary_header = "\n================= FINAL SUMMARY ================="
    log_message(summary_header, log_file)
    log_message(f"Total images processed: {processed_count}", log_file)
    
    log_message("\nNumber of images per category:", log_file)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if cat != "Unknown":
            log_message(f"  - {cat}: {count}", log_file)
            
    log_message(f"\nNumber of unknown reports: {unknown_count}", log_file)
    
    log_message(f"\nNewly created folders ({len(newly_created_folders)}):", log_file)
    if newly_created_folders:
        for folder in sorted(newly_created_folders):
            log_message(f"  - {folder}", log_file)
    else:
        log_message("  - None", log_file)
        
    log_message("=================================================", log_file)
    log_file.close()

if __name__ == "__main__":
    main()
