# Define variables
PYTHON=python
PIP=pip
REQUIREMENTS=requirements.txt
TRAIN_DATA=churn-bigml-80.csv
TEST_DATA=churn-bigml-20.csv
MODEL_FILE=modelRF.joblib

# Docker variables
IMAGE_NAME=rayen_lassoued_4ds1_mlops1
PORT=8090

# Default target (runs all tasks)
all: install prepare train evaluate run_api

# Install dependencies
install:
	$(PIP) install -r $(REQUIREMENTS)

# Prepare data
prepare:
	@echo "Preparing data..."
	$(PYTHON) -c "from model_pipeline import prepare_data; prepare_data('$(TRAIN_DATA)', '$(TEST_DATA)')"
	@echo "Data preparation completed."

# Start MLflow Server
start_mlflow:
	@echo "Starting MLflow Server..."
	nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
	@echo "MLflow is running at http://localhost:5000"

# Train and save model
train:
	@echo "Training the model..."
	$(PYTHON) main.py
	@echo "Model training completed and saved as $(MODEL_FILE)."

# Evaluate model
evaluate:
	@echo "Evaluating the model..."
	$(PYTHON) -c "from model_pipeline import evaluate_model, prepare_data; import joblib; X_train, y_train, X_test, y_test = prepare_data('$(TRAIN_DATA)', '$(TEST_DATA)'); model = joblib.load('$(MODEL_FILE)'); metrics = evaluate_model(model, X_test, y_test); print('Evaluation Metrics:', metrics)"
	@echo "Model evaluation completed."

# Run API
run_api:
	@echo "Starting FastAPI server..."
	$(PYTHON) -m uvicorn app:app --reload

# Start Elasticsearch and Kibana
start_elk:
	@echo "Starting Elasticsearch & Kibana..."
	sudo docker-compose up -d
	@echo "Elasticsearch running at http://localhost:9200"
	@echo "Kibana running at http://localhost:5601"

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f $(MODEL_FILE)
	@echo "Cleanup completed."
	
# Build Docker image
docker_build:
	@echo "Building Docker image..."
	sudo docker build -t $(IMAGE_NAME) .
	@echo "Docker image built successfully."

# Run Docker container
docker_run:
	@echo "Running Docker container..."
	sudo docker run -d -p $(PORT):8000 $(IMAGE_NAME)
	@echo "Container is running at http://localhost:$(PORT)/docs"

# Push image to Docker Hub
docker_push:
	@echo "Pushing image to Docker Hub..."
	sudo docker login
	sudo docker tag $(IMAGE_NAME) rayenlassoued/$(IMAGE_NAME):latest
	sudo docker push rayenlassoued/$(IMAGE_NAME):latest
	@echo "Image pushed to Docker Hub."

# Run all Docker tasks
docker_all: docker_build docker_run docker_push

