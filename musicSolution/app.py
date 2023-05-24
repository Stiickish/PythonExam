import streamlit as st
from pathlib import Path
import scipy
import pandas as pd
import implicit
import matplotlib.pyplot as plt

from data import load_user_artists, ArtistRetriever
from recommender import ImplicitRecommender

# Here user-artist matrix is loaded
user_artist_matrix = load_user_artists(Path("../musicSolution/lastfmdata/user_artists.dat"))

# Instantiate artist retriever:
artist_retriever = ArtistRetriever()
artist_retriever.load_artists(Path("../musicSolution/lastfmdata/artists.dat"))

def main():
    st.title("Music Solution 2000")

    # Hent User-id via input feltet
    user_id = st.number_input("User ID (min:2, max:2100)", min_value=2, max_value=2100, value=2, step=1)

    # Hent user-input for de tre ting der skal med i algoritmen (factors,  iterations og regularization)
    factors = st.number_input("Factors", value=50)
    iterations = st.number_input("Iterations", value=10)
    regularization = st.number_input("Regularization", value=0.01)

    # Button som kan starte algoritmen:
    execute_algorithm = st.button("Execute Algorithm")

    if execute_algorithm:
        # Instantiate Alternating Least Square med implicit hvor der benyttes brugerens input-værdier:
        implicit_model = implicit.als.AlternatingLeastSquares(
            factors=factors, iterations=iterations, regularization=regularization
        )

        # Instantiate recommender, fit, and recommend:
        recommender = ImplicitRecommender(artist_retriever, implicit_model)
        recommender.fit(user_artist_matrix)
        artists, scores = recommender.recommend(user_id, user_artist_matrix, n=5)

        # Laves to kolonner for at vise anbefalingerne side om side:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 5 recommendations for user ID: " + str(user_id))
            for artist, score in zip(artists, scores):
                st.write(f"{artist}: {score}")

        with col2:
            # Plot-tegningen er kolonne nummer 2
            fig, ax = plt.subplots(figsize=(8, 6))
            for artist, score in zip(artists, scores):
                ax.bar(artist, score)

            # Sæt labels for akserne og titlerne.
            ax.set_xlabel("Artist")
            ax.set_ylabel("Score")
            ax.set_title("Top 5 recommendations")
            ax.set_xticklabels(artists, rotation=45)

            # Vis plot-tegningen
            st.pyplot(fig)


if __name__ == "__main__":
    main()
