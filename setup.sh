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

# Ensure distutils is available (assuming it's installed)
# if ! python3 -c "import distutils" &> /dev/null
# then
#    apt-get install -y python3-distutils
# fi

# Install wheel to help with building packages
pip install wheel

# Install numpy (try a compatible version if necessary)
pip install numpy==1.19.5  # Or the version specified in requirements.txt

# Install Python dependencies from requirements file
pip install -r /mount/src/steamlit-bcd-app/requirements.txt

# Fix for distutils module missing error (commented out as distutils should be installed)
# python3 -m ensurepip --upgrade

# Try installing numpy 1.19.5 again (commented out as a different version is recommended)
# pip install numpy==1.19.5
