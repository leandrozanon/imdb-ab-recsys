"""
Recommendation System Module A.

This module prepares data, loads a KNN model, and generates predictions for
movie recommendations using user ratings.
"""

import joblib
import pandas as pd
import streamlit as st
from utils import descriptive_info, show_carousel
from sklearn.neighbors import NearestNeighbors


def load_data(file_path='./src/data/ratings_sample.csv'):
    """
    Load the MovieLens ratings dataset.
    
    Args:
        file_path (str): Path to the ratings CSV file.
        
    Returns:
        surprise.dataset.DatasetAutoFolds: Dataset ready for training.
    """
    # Load data
    ratings_df = pd.read_csv(file_path)
    
    return ratings_df

def dataprep_model_a(ratings_df):
    """
    Prepares the user-movie rating matrix for the KNN model.

    Args:
        ratings_df (pd.DataFrame): DataFrame containing user ratings.

    Returns:
        pd.DataFrame: Pivoted user-item matrix.
    """
    user_movie_matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        aggfunc='max'
    ).fillna(0)
    return user_movie_matrix


@st.cache_resource
def load_model_a():
    """
    Loads the pre-trained KNN model from disk.

    Returns:
        sklearn.neighbors.KNeighborsClassifier: Loaded model.
    """
    with open("./src/models/model_a.pkl", "rb") as file:
        model = joblib.load(file)
    return model

def train_model_a(ratings_df):
    """
    Trains a KNN model for movie recommendations.

    Args:
        ratings_df (pd.DataFrame): DataFrame containing user ratings.

    Returns:
        sklearn.neighbors.KNeighborsClassifier: Trained model.
    """
    user_movie_matrix = dataprep_model_a(ratings_df)
    model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
    model.fit(user_movie_matrix)
    joblib.dump(model, '../models/model_a.pkl')

def predict_model_a(model, user_id):
    """
    Predicts movie recommendations for a given user.

    Args:
        model: Trained KNN model.
        user_id (int): The ID of the user for whom recommendations are made.

    Returns:
        list: Top 5 recommended movie titles.
    """
    try:
        ratings_df = pd.read_csv('./src/data/ratings_sample.csv')
        user_movie_matrix = dataprep_model_a(ratings_df)

        # Find nearest neighbors for the user
        _, indices = model.kneighbors(
            [user_movie_matrix.loc[user_id].squeeze()]
        )

        st.write(f"Nearest neighbors: {indices[0][1]}")
        recommendations = descriptive_info(indices[0][1])

        return recommendations.sort_values(
            'rating', ascending=False).head(5)['title'].tolist()

    except KeyError:
        return {"Error": "User ID not found in dataset."}
    except Exception as e:
        return {"Error": f"An unexpected error occurred: {e}"}
    
def route_a(selected_id):
    try:
        model = load_model_a()
        recs = predict_model_a(model, selected_id)
        show_carousel(recs)  
    except:
        st.write("Fail load pretrained model, training model...")
        ratings_df = load_data()
        train_model_a(ratings_df)
        st.write("Model A trained and load successfully!")
        model = load_model_a()
        recs = predict_model_a(model, selected_id)
        show_carousel(recs)
