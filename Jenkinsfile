pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                script {
                    checkout scm
                }
            }
        }
        stage('Setup Python') {
            steps {
                script {
                    sh 'python -m pip install --upgrade pip'
                    sh 'pip install -r requirements.txt'
                }
            }
        }
        stage('Lint') {
            steps {
                script {
                    sh 'flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics'
                    sh 'flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics'
                }
            }
        }
        stage('Test') {
            steps {
                script {
                    sh 'pytest'
                }
            }
        }
    }
}
