import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import show_carousel


def load_data(movies_path='./src/data/movies.csv', ratings_path='./src/data/ratings_sample.csv'):
    """
    Load movie metadata and ratings dataset.

    Args:
        movies_path (str): Path to the movies CSV file.
        ratings_path (str): Path to the ratings CSV file.

    Returns:
        movies_df (DataFrame): DataFrame with movie metadata.
        ratings_df (DataFrame): DataFrame with user ratings.
    """
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    return movies_df, ratings_df


def train_tfidf_model(movies_df):
    """
    Train a TF-IDF model using the movie genres.

    Args:
        movies_df (DataFrame): DataFrame containing movie metadata.

    Returns:
        cosine_sim (ndarray): Cosine similarity matrix for movies.
        indices (Series): Mapping from movie titles to indices.
    """
    # Fill missing genres with empty strings
    movies_df['genres'] = movies_df['genres'].fillna('')

    # TF-IDF Vectorization on genres
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

    # Compute cosine similarity between movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Map movie titles to indices
    indices = pd.Series(movies_df.index, index=movies_df['movieId']).drop_duplicates()

    # Save the model to disk
    joblib.dump((cosine_sim, indices), './src/models/model_b.pkl')

    return cosine_sim, indices


def load_model_b():
    """
    Load the trained TF-IDF model from disk.

    Returns:
        cosine_sim (ndarray): Cosine similarity matrix.
        indices (Series): Mapping from movie IDs to indices.
    """
    with open("./src/models/model_b.pkl", "rb") as file:
        cosine_sim, indices = joblib.load(file)
    return cosine_sim, indices


def predict_model_b(user_id, ratings_df, movies_df, cosine_sim, indices, n=5):
    """
    Get top N movie recommendations for a specific user based on TF-IDF similarity.

    Args:
        user_id (int): ID of the user for whom to make recommendations.
        ratings_df (DataFrame): DataFrame with user ratings.
        movies_df (DataFrame): DataFrame with movie metadata.
        cosine_sim (ndarray): Cosine similarity matrix.
        indices (Series): Mapping from movie IDs to indices.
        n (int): Number of top recommendations to return.

    Returns:
        list: List of recommended movie IDs.
    """
    # Get the user's highest-rated movie
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    if user_ratings.empty:
        return ["No ratings found for this user."]

    top_rated_movie_id = user_ratings.loc[user_ratings['rating'].idxmax()]['movieId']
    idx = indices.get(top_rated_movie_id)

    if idx is None:
        return ["Top-rated movie not found in movie dataset."]

    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of the most similar movies (excluding the rated one)
    sim_indices = [i[0] for i in sim_scores[1:n+1]]

    # Return the recommended movie IDs
    recommended_movie_ids = movies_df.iloc[sim_indices]['movieId'].tolist()

    return recommended_movie_ids


def route_b(selected_id):
    try:
        # Load pre-trained model
        cosine_sim, indices = load_model_b()
        
        # Load data for prediction
        movies_df, ratings_df = load_data()
        
        # Get recommendations
        recs = predict_model_b(selected_id, ratings_df, movies_df, cosine_sim, indices)
        show_carousel(recs)

    except Exception:
        st.write("Failed to load pretrained model. Training model...")
        
        # Load data and train model
        movies_df, ratings_df = load_data()
        cosine_sim, indices = train_tfidf_model(movies_df)
        
        st.write("Model B trained and loaded successfully!")
        
        # Get recommendations
        recs = predict_model_b(selected_id, ratings_df, movies_df, cosine_sim, indices)
        show_carousel(recs)
        st.write(recs)