import pandas as pd
import mlflow
import mlflow.sklearn
import logging
from elasticsearch import Elasticsearch
from model_pipeline import prepare_data, train_and_save_model, evaluate_model  # Import your pipeline functions
from mlflow.tracking import MlflowClient  # Import the MlflowClient to interact with the Model Registry

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Set the tracking URI for MLflow

# Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlflow_logger")

# Fonction pour envoyer des logs à Elasticsearch
def log_to_elasticsearch(index, log_data):
    es.index(index=index, document=log_data)

def main():
    mlflow.set_experiment("Customer Churn Prediction")  # Set experiment name

    train_path = "churn-bigml-80.csv"  # Replace with your actual train data path
    test_path = "churn-bigml-20.csv"    # Replace with your actual test data path
    model_filename = "modelRF.joblib"

    X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)

    if X_train is None or y_train is None:
        logger.error("Data preparation failed. Exiting.")
        log_to_elasticsearch("mlflow-logs", {"message": "Data preparation failed", "status": "error"})
        return

    with mlflow.start_run():  # Start MLflow tracking

        # Log data shape
        mlflow.log_param("train_data_size", X_train.shape[0])
        mlflow.log_param("test_data_size", X_test.shape[0])

        log_to_elasticsearch("mlflow-logs", {"message": "Training started", "train_size": X_train.shape[0], "test_size": X_test.shape[0], "status": "running"})

        # Train and save the model
        trained_model = train_and_save_model(X_train, y_train, X_test, y_test, model_filename)

        if trained_model:
            if X_test is not None and y_test is not None:
                # Evaluate the model
                metrics = evaluate_model(trained_model, X_test, y_test)
                print("Evaluation Metrics:", metrics)

                # Log metrics
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("precision", metrics["precision"])
                mlflow.log_metric("recall", metrics["recall"])
                mlflow.log_metric("f1_score", metrics["f1_score"])

                # Log the model
                mlflow.sklearn.log_model(trained_model, "random_forest_model")

                # Register the model in the Model Registry
                client = MlflowClient()  # MLflow Client to interact with the Model Registry
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"  # Model URI
                
                # Create registered model if it does not exist
                try:
                    client.create_registered_model("random_forest_model")
                except Exception as e:
                    logger.info("Model already registered, skipping registration.")

                # Register a new version of the model
                model_version = client.create_model_version("random_forest_model", model_uri, "v1")
                logger.info(f"Model version {model_version.version} registered successfully.")

                log_to_elasticsearch("mlflow-logs", {
                    "message": "Model trained and versioned successfully",
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "status": "success"
                })

            else:
                logger.warning("No test data provided for evaluation.")
                log_to_elasticsearch("mlflow-logs", {"message": "No test data available", "status": "warning"})
        else:
            logger.error("Model training failed.")
            log_to_elasticsearch("mlflow-logs", {"message": "Model training failed", "status": "error"})

if __name__ == "__main__":
    main()

