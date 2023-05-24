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
    st.header("More iterations = more precise algorithm.\nHowever the perfomance is affected too. Recommended max iterations=50")

    # Get user input for factors, iterations, and regularization
    factors = st.number_input("Factors", value=50)
    iterations = st.number_input("Iterations", value=10)
    regularization = st.number_input("Regularization", value=0.01)

    # Create a button to execute the algorithm
    execute_algorithm = st.button("Execute Algorithm")

    if execute_algorithm:
        # Instantiate Alternating Least Square with implicit using user input values
        implicit_model = implicit.als.AlternatingLeastSquares(
            factors=factors, iterations=iterations, regularization=regularization
        )

        # Instantiate recommender, fit, and recommend:
        recommender = ImplicitRecommender(artist_retriever, implicit_model)
        recommender.fit(user_artist_matrix)
        artists, scores = recommender.recommend(2, user_artist_matrix, n=5)

        # Display recommendations and plot side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 5 recommendations for you")
            for artist, score in zip(artists, scores):
                st.write(f"{artist}: {score}")

        with col2:
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            for artist, score in zip(artists, scores):
                ax.bar(artist, score)

            # Set up the plot properties
            ax.set_xlabel("Artist")
            ax.set_ylabel("Score")
            ax.set_title("Top 5 recommendations")
            ax.set_xticklabels(artists, rotation=45)

            # Display the plot
            st.pyplot(fig)


if __name__ == "__main__":
    main()
