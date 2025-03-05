from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from model_pipeline import prepare_data, train_and_save_model

# Load the trained model
MODEL_PATH = "modelRF.joblib"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

app = FastAPI()

# Define request body for prediction
class PredictionInput(BaseModel):
    data: dict  # Expecting a dictionary of feature values

# Define request body for retraining
class RetrainInput(BaseModel):
    train_path: str
    test_path: str

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data.data])

    # Load the model's expected feature names
    model_features = joblib.load("model_features.joblib")  # Ensure this file exists

    # Ensure input matches model training features
    missing_cols = set(model_features) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Fill missing one-hot encoded columns with 0

    input_df = input_df[model_features]  # Reorder columns to match model training

    # Perform prediction
    try:
        prediction = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": prediction.tolist()}



@app.post("/retrain")
def retrain(input_data: RetrainInput):
    global model
    
    # Prepare data and retrain model
    X_train, y_train, X_test, y_test = prepare_data(input_data.train_path, input_data.test_path)
    if X_train is None or y_train is None:
        raise HTTPException(status_code=500, detail="Data preparation failed.")
    
    model = train_and_save_model(X_train, y_train, X_test, y_test, MODEL_PATH)
    return {"message": "Model retrained successfully."}

@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI ML Service!"}
