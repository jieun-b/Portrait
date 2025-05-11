#!/bin/bash

# Conda 환경 생성
ENV_NAME="portrait"
PYTHON_VERSION="3.11"

echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Conda 환경 활성화
source activate $ENV_NAME

# 패키지 설치
echo "Installing packages..."
pip install -r requirements.txt

echo "Environment setup completed successfully!"
