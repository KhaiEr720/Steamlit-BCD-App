#!/bin/bash

# Option 1: Using Conda
echo "Creating a new conda environment with Python 3.12..."
conda create -n my-env python=3.12 -y
conda activate my-env
echo "Installing dependencies..."
conda install numpy matplotlib streamlit pandas seaborn plotly scikit-learn -y

# Option 2: Using Pyenv
echo "Installing Python 3.11.4 using pyenv..."
pyenv install 3.11.4
echo "Setting global Python version to 3.11.4..."
pyenv global 3.11.4

# Continue with the rest of your setup
echo "Setup complete."

# Create required directories
mkdir -p ~/.streamlit/

# Create credentials file
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

# Create config file
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml

# Upgrade setuptools to ensure distutils is available
pip install --upgrade setuptools

# Install numpy
pip install numpy==1.26.0

# Install Python dependencies from requirements file
pip install -r /mount/src/steamlit-bcd-app/requirements.txt
