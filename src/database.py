import os
import json
from datetime import datetime
from supabase import create_client, Client

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase_client: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
else:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not found in environment variables. Database logging will be disabled.")

def init_db():
    """
    Initializes the database and validates the connection by performing a lightweight query.
    """
    if not supabase_client:
        print("Warning: Supabase client is not initialized. Skipping DB verification.")
        return
    
    try:
        # Perform a lightweight select to verify connection and table existence
        supabase_client.table("reports").select("id").limit(1).execute()
        print("Database connection verified. 'reports' table is accessible in Supabase.")
    except Exception as e:
        print(f"Warning: Failed to connect/verify 'reports' table in Supabase: {e}")
        print("Please ensure the 'reports' table has been created using the SQL script in DEPLOYMENT.md.")

def save_report_log(image_path: str, ocr_text: str, predicted_category: str, confidence: float, matched_keywords: list) -> int:
    """
    Saves a report record to the Supabase database and returns the generated report ID.
    """
    if not supabase_client:
        print("Warning: Supabase not configured. Log not saved to database.")
        return 0
        
    upload_date = datetime.now().isoformat()
    matched_keywords_json = json.dumps(matched_keywords)
    filename = os.path.basename(image_path)
    
    data = {
        "filename": filename,
        "upload_date": upload_date,
        "image_url": image_path,
        "ocr_text": ocr_text,
        "predicted_category": predicted_category,
        "confidence": float(confidence),
        "matched_keywords": matched_keywords_json
    }
    
    try:
        response = supabase_client.table("reports").insert(data).execute()
        # The Supabase Python SDK returns a response object with a .data attribute
        if response.data and len(response.data) > 0:
            inserted_id = response.data[0].get("id")
            print(f"Successfully saved log to Supabase. Record ID: {inserted_id}")
            return inserted_id
        else:
            print("Warning: Supabase insert returned empty data response.")
            return 0
    except Exception as e:
        print(f"Error saving log to Supabase: {e}")
        return 0

def update_report_feedback(report_id: int, user_confirmation: str, final_correct_label: str) -> bool:
    """
    Updates the feedback columns for the given report_id in Supabase.
    """
    if not supabase_client:
        print("Warning: Supabase not configured. Feedback not updated in database.")
        return False
        
    data = {
        "user_confirmation": user_confirmation,
        "final_correct_label": final_correct_label
    }
    
    try:
        response = supabase_client.table("reports").update(data).eq("id", report_id).execute()
        if response.data and len(response.data) > 0:
            print(f"Successfully updated feedback for Record ID {report_id} in Supabase.")
            return True
        else:
            print(f"Warning: No record found with ID {report_id} in Supabase to update.")
            return False
    except Exception as e:
        print(f"Error updating feedback in Supabase: {e}")
        return False
