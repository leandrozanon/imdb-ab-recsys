"""
Main Application for A/B Testing with Machine Learning Models.

This app allows users to select a user ID, generate personalized movie
recommendations using different algorithms, and display results in an interactive
Streamlit interface.
"""

import pandas as pd
import streamlit as st
from utils import descriptive_info, sample_group
from recommend_a import route_a
from recommend_b import route_b


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
        # Route A
        if group == 'A':
            route_a(selected_id)
        # Route B
        elif group == 'B':
            route_b(selected_id)

