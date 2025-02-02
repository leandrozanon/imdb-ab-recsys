import pandas as pd
import streamlit as st
import logging
from utils import descriptive_info, sample_group
from recommend_a import route_a
from recommend_b import route_b

# Set up logging configuration
logging.basicConfig(
    filename='./src/logs/app.log',  # Log file location
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_route_selection(group):
    """
    Log the route selection for A/B testing.

    Args:
        group (str): Selected group for the user (A or B).
    """
    logging.info(f"Route selected: {group}")

if __name__ == "__main__":
    st.title("A/B Testing with Machine Learning Models")

    ratings = pd.read_csv('./src/data/ratings_sample.csv')
    selected_id = st.selectbox(
        "Select a userId:",
        sorted(ratings["userId"].unique())
    )

    descriptive_info(selected_id)

    if st.button("Get recs"):
        group = sample_group()
        
        # Log the route selection (A or B)
        log_route_selection(group)
        
        # Route A
        if group == 'A':
            route_a(selected_id)
        # Route B
        elif group == 'B':
            route_b(selected_id)

