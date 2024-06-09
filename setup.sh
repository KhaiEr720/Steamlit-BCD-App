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


pip install --upgrade setuptools
python -c "import distutils"

# Install numpy
pip install numpy==1.19.3

# Install Python dependencies from requirements file
pip install -r /mount/src/steamlit-bcd-app/requirements.txt
