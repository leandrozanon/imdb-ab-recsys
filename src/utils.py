"""
Utility functions for processing movie data and displaying information with Streamlit.
"""

import random
import pandas as pd
import streamlit as st
from imdb import IMDb


def get_movie_info_from_user_id(user_id):
    """
    Retrieve movie information for a specific user ID.

    Args:
        user_id (int): The ID of the user.

    Returns:
        DataFrame: Movies watched by the user along with ratings and genres.
    """
    movies = pd.read_csv('./src/data/movies.csv')
    ratings = pd.read_csv('./src/data/ratings_sample.csv')

    # Filter ratings for the specified user and pivot the table
    user_ratings = (
        ratings[(ratings['userId'] == user_id) & (ratings['rating'] > 0)]
        .pivot_table(index=['userId'], columns='movieId', values='rating', aggfunc='max')
        .fillna(0)
    ).squeeze()

    watched_movie_ids = sorted(user_ratings.index.tolist())
    selected_movies_df = (
        movies[movies['movieId'].isin(watched_movie_ids)]
        .merge(ratings[ratings['userId'] == user_id], on='movieId')
        [['userId', 'movieId', 'title', 'genres', 'rating']]
    )

    return selected_movies_df


def descriptive_info(user_id):
    """
    Display movie information for a given user in Streamlit.

    Args:
        user_id (int): The ID of the user.

    Returns:
        DataFrame: Sorted movie details for the user.
    """
    selected_movies_df = get_movie_info_from_user_id(user_id)

    st.write("### Selected User Details")
    st.dataframe(selected_movies_df.sort_values('rating', ascending=False),
                 use_container_width=True,
                 hide_index=True)

    return selected_movies_df.sort_values('rating', ascending=False)


def show_carousel(movie_names):
    """
    Display a carousel of movie images in Streamlit.

    Args:
        movie_names (list): List of movie titles.

    Returns:
        dict: A dictionary mapping movie IDs to their cover image URLs.
    """
    def get_images(movie_id_list):
        links = pd.read_csv('./src/data/links.csv', dtype={'imdbId': str})
        image_list = {}

        for movie_id in sorted(movie_id_list):
            ia = IMDb()
            imdb_id = links[links['movieId'] == movie_id]['imdbId'].iloc[0]
            imdb_movie = ia.get_movie(str(imdb_id).zfill(7))
            image_list[movie_id] = imdb_movie.get(
                'cover url', 'Image not available')

        return image_list

    movies = pd.read_csv(
        './src/data/movies.csv')[['movieId', 'title', 'genres']]
    movie_ids = movies[movies['title'].isin(movie_names)]['movieId'].tolist()
    image_list = get_images(movie_ids)

    for movie_id, image_url in image_list.items():
        movie_name = movies[movies['movieId'] == movie_id]['title'].iloc[0]
        movie_genre = movies[movies['movieId'] == movie_id]['genres'].iloc[0]
        st.image(image_url,
                 use_container_width=False,
                 caption=f"{movie_name} - Genre: {movie_genre}")

    return image_list


def sample_group():
    """
    Randomly assign a user to group 'A' or 'B'.

    Returns:
        str: Group assignment ('A' or 'B').
    """
    return random.choice(['A', 'B'])
