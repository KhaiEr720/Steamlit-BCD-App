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
  build-essential

# Install distutils
apt-get install -y python3-apt

# Check and install pip for Python 3
if ! command -v pip3 &> /dev/null
then
    apt-get install -y python3-pip
fi

# Install Python dependencies
pip install -r requirements.txt

# Check for missing modules and install them
if ! python3 -c "import distutils" &> /dev/null
then
    apt-get install -y python3-distutils
fi

# Additional steps to ensure numpy is installed properly
pip install numpy==1.19.5 --no-cache-dir

# Finally, install the rest of the dependencies
pip install -r requirements.txt
