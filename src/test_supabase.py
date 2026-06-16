import os
import sys
from dotenv import load_dotenv

# Ensure project src directory is in path for modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load variables from .env
load_dotenv()

import database

def main():
    print("=== Supabase Connection & Insertion Test ===")
    
    # Check if variables are loaded
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    print(f"SUPABASE_URL: {url}")
    if not url or not key:
        print("[ERROR] SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not defined in the environment.")
        sys.exit(1)
        
    print("Initializing connection...")
    database.init_db()
    
    print("\nAttempting to insert a sample test report...")
    sample_filename = "test_sample_report.png"
    sample_ocr = "Patient Name: Jane Doe. CBC report. wbc count: 7000 cells/cumm. RBC count: 4.5 million/cumm."
    sample_category = "cbc"
    sample_confidence = 0.95
    sample_keywords = ["wbc", "rbc", "complete blood count"]
    
    report_id = database.save_report(
        filename=sample_filename,
        ocr_text=sample_ocr,
        predicted_category=sample_category,
        confidence=sample_confidence,
        matched_keywords=sample_keywords
    )
    
    if report_id > 0:
        print(f"[SUCCESS] Sample report inserted with ID: {report_id}")
    else:
        print("[ERROR] Could not insert sample report.")
        sys.exit(1)
        
    print("\nAttempting to query the inserted record...")
    if database.supabase_client:
        try:
            response = database.supabase_client.table("reports").select("*").eq("id", report_id).execute()
            if response.data and len(response.data) > 0:
                record = response.data[0]
                print("[SUCCESS] Retrieved record details:")
                print(f"  ID:                 {record.get('id')}")
                print(f"  Filename:           {record.get('filename')}")
                print(f"  Predicted Category: {record.get('predicted_category')}")
                print(f"  Confidence:         {record.get('confidence')}")
                print(f"  Created At:         {record.get('created_at')}")
            else:
                print("[ERROR] Record was inserted but could not be found.")
                sys.exit(1)
        except Exception as e:
            print(f"Error querying table: {e}")
            sys.exit(1)
            
    print("\nAttempting to update user feedback on the record...")
    update_success = database.update_feedback(
        report_id=report_id,
        user_confirmation="correct",
        final_correct_label="cbc"
    )
    
    if update_success:
        print("[SUCCESS] Feedback updated in the database.")
    else:
        print("[ERROR] Could not update feedback.")
        sys.exit(1)
        
    print("\n=== All Database Verification Checks Completed Successfully ===")

if __name__ == "__main__":
    main()
