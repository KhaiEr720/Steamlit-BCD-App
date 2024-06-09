#!/bin/bash

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
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml

# Upgrade setuptools to ensure distutils is available
pip install --upgrade setuptools

# Install Python dependencies from requirements file
pip install -r /mount/src/steamlit-bcd-app/requirements.txt
