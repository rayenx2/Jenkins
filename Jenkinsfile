pipeline {
    agent any

    environment {
        PYTHON = 'python'
        PIP = 'pip'
        REQUIREMENTS = 'requirements.txt'
        TRAIN_DATA = 'churn-bigml-80.csv'
        TEST_DATA = 'churn-bigml-20.csv'
        MODEL_FILE = 'modelRF.joblib'
    }

    stages {
        stage('Clone Repository') {
            steps {
                echo 'Cloning the repository...'
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo 'Setting up Python virtual environment...'
                sh 'python3 -m venv venv'
                sh 'source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt'
            }
        }

        stage('Prepare Data') {
            steps {
                echo 'Preparing data...'
                sh 'source venv/bin/activate && python -c "from model_pipeline import prepare_data; prepare_data(\'${TRAIN_DATA}\', \'${TEST_DATA}\')"'
            }
        }

        stage('Start MLflow Server') {
            steps {
                echo 'Starting MLflow Tracking Server...'
                sh 'nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &'
            }
        }

        stage('Train Model') {
            steps {
                echo 'Training the model...'
                sh 'source venv/bin/activate && python main.py'
            }
        }

        stage('Evaluate Model') {
            steps {
                echo 'Evaluating the model...'
                sh 'source venv/bin/activate && python -c "from model_pipeline import evaluate_model, prepare_data; import joblib; X_train, y_train, X_test, y_test = prepare_data(\'${TRAIN_DATA}\', \'${TEST_DATA}\'); model = joblib.load(\'${MODEL_FILE}\'); metrics = evaluate_model(model, X_test, y_test); print(\'Evaluation Metrics:\', metrics)"'
            }
        }
    }

    post {
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline failed! Check logs for errors.'
        }
    }
}

