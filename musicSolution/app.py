import csv

import streamlit as st
from pathlib import Path
import scipy
import pandas as pd
import implicit
import matplotlib.pyplot as plt
import numpy as np
import time
from data import load_user_artists, ArtistRetriever
from recommender import ImplicitRecommender
from streamlit import session_state

# Here user-artist matrix is loaded
user_artist_matrix = load_user_artists(Path("../musicSolution/lastfmdata/user_artists.csv"))
max_user_id = user_artist_matrix.shape[0] - 1 #Vi skal finde et max_id som vi kan bruge i funktioner.
total_number_of_users = user_artist_matrix.shape[0]
print(total_number_of_users)

# Instantiate artist retriever:
artist_retriever = ArtistRetriever()
artist_retriever.load_artists(Path("../musicSolution/lastfmdata/artists.dat"))


def main():
    st.title("Music Solution 2000")

    # Determine the maximum user ID based on the loaded user-artist matrix
    # max_user_id = user_artist_matrix.shape[0] - 1
    file_path = "./lastfmdata/user_artists.csv"
    # Nu skal vi kontrollere hvad max_user_id er ud fra sidste linje i filen.
    if Path(file_path).is_file():
        with open(file_path, "r") as csvfile:
            lines = csvfile.readlines()
            if lines:
                last_line = lines[-1].strip().split("\t")
                max_user_id = int(last_line[0])

    if "input_user_artists" not in session_state:
        session_state["input_user_artists"] = []

    # Hent User-id via input feltet
    user_id = st.number_input(f"User ID (min:2, max:{max_user_id})", min_value=2, max_value=max_user_id, value=2, step=1)

    # Hent user-input for de tre ting der skal med i algoritmen (factors,  iterations og regularization)
    factors = st.number_input("Factors", value=50)
    iterations = st.number_input("Iterations", value=10)
    regularization = st.number_input("Regularization", value=0.01)

    # Button som kan starte algoritmen:
    execute_algorithm = st.button("Execute Algorithm")

    if execute_algorithm:
        update_user_artists(user_id, session_state.input_user_artists, user_artist_matrix)
        # Append the user's favorite bands to user_artist_matrix
        # update_user_artists(user_id, session_state.input_user_artists, user_artist_matrix)

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

    st.header(
        "Instead of getting recommendations from another user, you can get your own recommendations. Write the name of at least 5 artists and maximum 10 to get a precise recommendation.")
    input_user_artist_likes = st.text_input("Write the name of an artist you like and click 'Add'")
    if st.button("Add"):
        if len(session_state.input_user_artists) >= 10:
            st.write("You have reached the maximum limit of 10 artists.")
        else:
            artist_already_added = False
            for artist in session_state.input_user_artists:
                if artist.lower() == input_user_artist_likes.lower():
                    artist_already_added = True
                    break

            if artist_already_added:
                st.write("Artist is already in the list!")
                st.text("Current artists: " + ", ".join(
                    session_state.input_user_artists))  # Display the current list of input artists

            else:
                # Hver gang brugeren inputter en artist kører vi lige igennem artists.dat for at se om artisten findes i datasættet.
                artist_id = get_artist_id(input_user_artist_likes)
                if artist_id is not None:
                    session_state.input_user_artists.append(input_user_artist_likes)
                    st.write("Artist added successfully!")
                    # Hmmmm ... jeg vil gerne have ID med ud ... how do I do that?
                    artist_ids = [str(get_artist_id(artist)) for artist in session_state.input_user_artists]
                    artist_info = ", ".join(f"{artist} (ID: {artist_id})" for artist, artist_id in
                                            zip(session_state.input_user_artists, artist_ids))
                    st.text("Current artists: " + artist_info)
                else:
                    st.write("Artist not found in the database.")

    if len(session_state.input_user_artists) >= 5:
        execute_algorithm_with_user_data = st.button("Put list in database", key="user_input_button")

        if execute_algorithm_with_user_data:
            update_user_artists(user_id, session_state.input_user_artists, user_artist_matrix)

            # Gem listen i filen.
            with open(file_path, "a",
                      newline="") as csvfile:  # med "a" i stedet for "w" appendes der, så den ikke laver en ny liste hver gang.
                writer = csv.writer(csvfile, delimiter="\t")
                for i, artist in enumerate(session_state.input_user_artists, start=1):
                    artist_id = get_artist_id(artist)
                    if artist_id is not None:
                        weight = str(15000 - (1000 * (i - 1)) if i <= 10 else 5000)  # Convert weight to string
                        weight = int(weight.strip())  # Remove leading/trailing spaces and convert back to integer
                        writer.writerow([str(max_user_id + 1), str(artist_id).zfill(4),
                                         weight])  # Convert artist_id to a 4-digit string
            st.write("Your list was saved.")

            # Opdatér user_id, så der kan laves nye sammenligninger:
            max_user_id += 1
            # Reset the input_user_artists list
            session_state.input_user_artists = []

            # Rerun the script to reload the page after a short delay
            st.experimental_rerun()
def get_artist_id(artist_name):
    with open("../musicSolution/lastfmdata/artists.dat", "r", encoding="utf-8") as artists_file:
        for line in artists_file:
            artist_id, name, *_ = line.strip().split("\t")
            if name.lower() == artist_name.lower():
                return int(artist_id.lstrip(
                    "0"))  # med lstrip fjernes der hvad stringen begynder med, hvis det matcher det der indsættes ( her "0" ).
    return None


def update_user_artists(user_id, favorite_bands, user_artist_matrix):
    for band in favorite_bands:
        artist_id = get_artist_id(band)  # Retrieve the artist ID based on the band name
        if artist_id is not None:
            # Update the user_artist_matrix with the new entry
            user_artist_matrix[user_id, artist_id] = 1  # Set the weight as 1


if __name__ == "__main__":
    main()
