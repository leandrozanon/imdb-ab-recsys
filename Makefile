# Makefile for Movie Recommendation System with Apriori Algorithm

# Define paths
SRC_DIR = ./src

# Python environment
PYTHON = python3
PIP = pip3

# Default target
all: install

# Install the required Python packages
install:
	$(PIP) install -r requirements.txt

# Run the Streamlit app
run_app:
	$(PYTHON) -m streamlit run $(SRC_DIR)/app.py