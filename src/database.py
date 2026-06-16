import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client using Service Role Key for administrative access
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase_client: Client = None

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("Supabase client initialized successfully with Service Role Key.")
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
else:
    print("Warning: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in environment variables. Database logging will be disabled.")

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
        print("Please ensure the 'reports' table has been created in your Supabase database.")

def save_report(filename: str, ocr_text: str, predicted_category: str, confidence: float, matched_keywords: list, user_confirmation: str = None, final_correct_label: str = None) -> int:
    """
    Saves a report record to the Supabase database and returns the generated report ID.
    Note: Supabase table schema uses 'created_at' (automated) and has no 'image_url' or 'upload_date' columns.
    'matched_keywords' is stored as a native jsonb column (accepts list directly).
    """
    if not supabase_client:
        print("Warning: Supabase not configured. Log not saved to database.")
        return 0
    
    data = {
        "filename": filename,
        "ocr_text": ocr_text,
        "predicted_category": predicted_category,
        "confidence": float(confidence),
        "matched_keywords": matched_keywords, # jsonb column accepts list natively
        "user_confirmation": user_confirmation,
        "final_correct_label": final_correct_label
    }
    
    try:
        response = supabase_client.table("reports").insert(data).execute()
        if response.data and len(response.data) > 0:
            inserted_id = response.data[0].get("id")
            print(f"Successfully saved report to Supabase. Record ID: {inserted_id}")
            return inserted_id
        else:
            print("Warning: Supabase insert returned empty data response.")
            return 0
    except Exception as e:
        print(f"Error saving report to Supabase: {e}")
        return 0

def update_feedback(report_id: int, user_confirmation: str, final_correct_label: str) -> bool:
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
