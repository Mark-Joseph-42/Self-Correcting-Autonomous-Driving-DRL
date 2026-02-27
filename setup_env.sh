#!/bin/bash
# Create virtual environment
# Using /usr/bin/python3 (Python 3.9) to satisfy metadrive requirements (<3.12)
/usr/bin/python3 -m venv driving_env

# Activate virtual environment and install requirements
./driving_env/bin/pip install --upgrade pip
./driving_env/bin/pip install -r requirements.txt

echo "Environment setup complete. Activate with: source driving_env/bin/activate"
