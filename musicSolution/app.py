import streamlit as st
from pathlib import Path
import scipy
import pandas as pd
import implicit
import matplotlib.pyplot as plt

from data import load_user_artists, ArtistRetriever
from recommender import ImplicitRecommender


def main():
    st.title("Music Solution 2000")

    # Here user-artist matrix is loaded
    user_artist_matrix = load_user_artists(Path("../musicSolution/lastfmdata/user_artists.dat"))

    # Instantiate artist retriever:
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../musicSolution/lastfmdata/artists.dat"))

    # Instantiate Alternating Least Square with implicit
    implicit_model = implicit.als.AlternatingLeastSquares(factors=50, iterations=10, regularization=0.01)

    # Instantiate recommender, fit, and recommend:
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artist_matrix)
    artists, scores = recommender.recommend(2, user_artist_matrix, n=5)

    # Display recommendations:
    st.subheader("Top 5 recommendations for you")
    for artist, score in zip(artists, scores):
        st.write(f"{artist}: {score}")


if __name__ == "__main__":
    main()
