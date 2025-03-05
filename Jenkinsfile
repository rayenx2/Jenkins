pipeline {
    agent any

    environment {
        VENV = 'venv'  // Define the virtual environment directory
        PYTHON = 'python3'  // Ensure you are using python3
    }

    stages {
        stage('Checkout GIT') {
            steps {
                echo 'Pulling latest code...'
                git branch: 'main', url: 'https://github.com/rayenx2/Jenkins.git'
            }
        }

        stage('Debug') {
            steps {
                echo 'Current Directory:'
                sh 'pwd'
                echo 'Listing Files:'
                sh 'ls -la'
                echo 'Checking for necessary files:'
                script {
                    def reqFile = 'requirements.txt'
                    if (fileExists(reqFile)) {
                        echo "${reqFile} file found!"
                    } else {
                        error "${reqFile} file not found. Aborting build."
                    }
                    def makefile = 'Makefile'
                    if (fileExists(makefile)) {
                        echo "${makefile} file found!"
                    } else {
                        error "${makefile} file not found. Aborting build."
                    }
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing dependencies...'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Prepare Data') {
            steps {
                echo 'Preparing data...'
                sh '''
                    . venv/bin/activate
                    make prepare
                '''
            }
        }

        stage('Train Model') {
            steps {
                echo 'Training model...'
                sh '''
                    . venv/bin/activate
                    make train
                '''
            }
        }

        stage('Evaluate Model') {
            steps {
                echo 'Evaluating model...'
                sh '''
                    . venv/bin/activate
                    make evaluate
                '''
            }
        }

        // New stage to start FastAPI app for prediction
        stage('Start FastAPI App') {
            steps {
                echo 'Starting FastAPI app for prediction...'
                sh '''
                    . venv/bin/activate
                    make run_api
                '''
            }
        }

        // New stage to start MLflow server (tracking API and UI)
        stage('Start MLflow Server') {
            steps {
                echo 'Starting MLflow server...'
                sh '''
                    . venv/bin/activate
                    make start_mlflow
                '''
            }
        }


    }

    post {
        always {
            echo 'Cleaning workspace...'
            deleteDir()
        }
    }
}
