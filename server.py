from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
from pathlib import Path
import joblib
import pandas as pd

# Import our custom OpenCV script
from ml_pipeline.chromatic_extractor import extract_dominant_hex, heuristic_diagnosis

app = FastAPI()

# Enable CORS for Next.js UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_remediation_data():
    # Get the exact, absolute path of the folder where server.py lives
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "datasets" / "remediation.json"
    
    print(f"\n--- DEBUG: Looking for database at: {file_path} ---")
    
    if not file_path.exists():
        print("--- DEBUG ERROR: FILE NOT FOUND! Check your folder structure. ---")
        return {"error": "File not found"}
        
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print(f"--- DEBUG SUCCESS: Loaded keys: {list(data.keys())} ---\n")
            return data
    except json.JSONDecodeError:
        print("--- DEBUG ERROR: Your JSON file has a typo (missing comma or bracket). ---")
        return {"error": "Invalid JSON"}

REMEDIATION_DB = load_remediation_data()

# Load the Machine Learning Model
MODEL_PATH = Path(__file__).resolve().parent / "ml_pipeline" / "color_model.joblib"
if MODEL_PATH.exists():
    clf_model = joblib.load(MODEL_PATH)
    print(f"--- ML MODEL LOADED SUCCESSFULLY from {MODEL_PATH} ---")
else:
    clf_model = None
    print("--- WARNING: ML Model not found. Falling back to heuristics. ---")


@app.get("/")
def read_root():
    # This will output exact debug info directly to your browser
    return {
        "message": "TerraScope Engine is running",
        "database_keys_found": list(REMEDIATION_DB.keys()),
        "database_error": REMEDIATION_DB.get("error", "None - Database Loaded Successfully!"),
        "ml_model_status": "Loaded" if clf_model else "Missing - Using Heuristics"
    }

@app.post("/api/v1/diagnose")
async def diagnose_leaf(file: UploadFile = File(...)):
    # 1. Read the uploaded file into memory
    image_bytes = await file.read()
    
    # 2. Extract the true color data using OpenCV
    hex_code, rgb_values = extract_dominant_hex(image_bytes)
    
    # 3. Determine the class based on our trained ML model
    if clf_model:
        # Sklearn expects a 2D DataFrame for prediction to match the feature names
        features = pd.DataFrame([rgb_values], columns=["red", "green", "blue"])
        prediction_class = clf_model.predict(features)[0]
    else:
        # Fallback if the model fails to load
        prediction_class = heuristic_diagnosis(*rgb_values)
    
    # 4. Fetch the rich UI data
    diagnosis_data = REMEDIATION_DB.get(prediction_class, {"error": "Unknown diagnosis"})
    
    # 5. Inject the live extracted hex code so the UI can display it
    diagnosis_data["live_extracted_hex"] = hex_code
    diagnosis_data["prediction_method"] = "Machine Learning (Random Forest)" if clf_model else "Heuristics"

    return {
        "filename": file.filename, 
        "status": "Success",
        "results": diagnosis_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)