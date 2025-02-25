from fastapi import FastAPI, HTTPException, Depends
import joblib
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from model_pipeline import prepare_data, train_and_save_model

# Charger les variables d'environnement
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root@db:5432/predictions_db")
MODEL_PATH = "modelRF.joblib"
MODEL_FEATURES_PATH = "model_features.joblib"

# Connexion à la base de données
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Définition du modèle SQLAlchemy pour stocker les prédictions
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    result = Column(String)

# Créer la table si elle n'existe pas
Base.metadata.create_all(bind=engine)

# Charger le modèle
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

model = load_model()

# Initialiser l'API FastAPI
app = FastAPI()

# Définir les entrées pour la prédiction
class PredictionInput(BaseModel):
    data: dict

# Définir les entrées pour le retrain
class RetrainInput(BaseModel):
    train_path: str
    test_path: str

# Dépendance pour obtenir une session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint de prédiction
@app.post("/predict")
def predict(input_data: PredictionInput, db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Convertir les données en DataFrame
    input_df = pd.DataFrame([input_data.data])

    # Charger les features du modèle
    try:
        model_features = joblib.load(MODEL_FEATURES_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Feature names file not found.")

    # Ajuster les colonnes d'entrée
    missing_cols = set(model_features) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  
    input_df = input_df[model_features]  

    # Effectuer la prédiction
    try:
        prediction = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Sauvegarder la prédiction dans la base de données
    prediction_entry = Prediction(result=str(prediction[0]))
    db.add(prediction_entry)
    db.commit()
    db.refresh(prediction_entry)

    return {"prediction": prediction.tolist(), "saved_id": prediction_entry.id}

# Endpoint de retrain du modèle
@app.post("/retrain")
def retrain(input_data: RetrainInput):
    global model

    # Préparer les données et entraîner le modèle
    X_train, y_train, X_test, y_test = prepare_data(input_data.train_path, input_data.test_path)
    if X_train is None or y_train is None:
        raise HTTPException(status_code=500, detail="Data preparation failed.")

    model = train_and_save_model(X_train, y_train, X_test, y_test, MODEL_PATH)
    return {"message": "Model retrained successfully."}

# Endpoint par défaut
@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI ML Service!"}

