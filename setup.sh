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

# Update pip and setuptools
pip install --upgrade pip setuptools

# Install system dependencies
apt-get update && apt-get install -y \
  python3-distutils \
  python3-dev \
  build-essential \
  python3-apt \
  python3-pip

# Ensure distutils is available
if ! python3 -c "import distutils" &> /dev/null
then
    apt-get install -y python3-distutils
fi

# Install wheel to help with building packages
pip install wheel

# Install numpy separately to avoid issues with other dependencies
pip install numpy==1.19.5 --no-cache-dir

# Install Python dependencies from requirements file
pip install -r /mount/src/steamlit-bcd-app/requirements.txt

# Fix for distutils module missing error
python3 -m ensurepip --upgrade
