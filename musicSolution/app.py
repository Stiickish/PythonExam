import csv
import streamlit as st
from pathlib import Path
import implicit
import matplotlib.pyplot as plt
from data import load_user_artists, ArtistRetriever
from recommender import ImplicitRecommender
from streamlit import session_state

st.set_page_config(layout="wide")
col1, col2 = st.columns(2)

with col1:
    st.image("images/logo-no-background.png", width=300)

with col2:
    st.title("Music Solution 2000")
    info = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    In vitae elementum mi. Mauris sit amet tincidunt risus, vitae semper lorem.
    Ut et nisl dictum, sodales arcu quis, ultricies ex. Suspendisse hendrerit, neque vitae luctus bibendum, urna quam molestie velit, ac tincidunt metus mi ut massa. 
    Nam ac interdum neque. Sed feugiat velit velit, ut mattis nisi facilisis nec. 
    Nunc accumsan euismod diam. Curabitur commodo ex lobortis feugiat posuere. 
    Vivamus et turpis sed lorem tristique pharetra non sed nisl. Phasellus vitae pretium massa. Aliquam erat volutpat. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    """
    st.write(info)


    # Here user-artist matrix is loaded
user_artist_matrix = load_user_artists(Path("../musicSolution/lastfmdata/user_artists.dat"))

# Instantiate artist retriever:
artist_retriever = ArtistRetriever()
artist_retriever.load_artists(Path("../musicSolution/lastfmdata/artists.dat"))


def main():
    # Determine the maximum user ID based on the loaded user-artist matrix
    # max_user_id = user_artist_matrix.shape[0] - 1
    file_path = "./lastfmdata/user_artist_input_list.csv"
    # Nu skal vi kontrollere hvad max_user_id er ud fra sidste linje i filen.
    with open(file_path, "r") as csvfile:
        lines = csvfile.readlines()
        if lines:
            last_line = lines[-1].strip().split(" ")
            max_user_id = int(last_line[0])

    if "input_user_artists" not in session_state:
        session_state["input_user_artists"] = []

    # Hent User-id via input feltet
    user_id = st.number_input(f"User ID (min:2, max:{max_user_id})", min_value=2, max_value=max_user_id, value=2,
                              step=1)

    # Hent user-input for de tre ting der skal med i algoritmen (factors,  iterations og regularization)
    factors = st.number_input("Factors", value=50)
    iterations = st.number_input("Iterations", value=10)
    regularization = st.number_input("Regularization", value=0.01)

    # Button som kan starte algoritmen:
    execute_algorithm = st.button("Execute Algorithm")

    if execute_algorithm:
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

    st.subheader(
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
                session_state.input_user_artists.append(input_user_artist_likes)
                st.write("Artist added successfully!")
                st.text("Current artists: " + ", ".join(
                    session_state.input_user_artists))  # Display the current list of input artists

    if len(session_state.input_user_artists) >= 5:
        execute_algorithm_with_user_data = st.button("Execute Algorithm on your data", key="user_input_button")
        if execute_algorithm_with_user_data:

            # Gem listen i filen.
            with open(file_path, "a",
                      newline="") as csvfile:  # med "a" i stedet for "w" appendes der, så den ikke laver en ny liste hver gang.
                writer = csv.writer(csvfile, delimiter=" ")
                for i, artist in enumerate(session_state.input_user_artists, start=1):
                    artist_id = i if i <= 10 else 10  # Limit artistID to a maximum of 10
                    weight = 15000 - (1000 * (i - 1)) if i <= 10 else 5000  # Calculate weight based on position
                    writer.writerow([max_user_id + 1, artist_id, weight])

            st.write("Your list was saved.")

            # Opdatér user_id, så der kan laves nye sammenligninger:
            max_user_id += 1


# def update_user_artists(user_id, favorite_bands, user_artist_matrix):
#     for band in favorite_bands:
#         artist_id = get_artist_id(band)  # Retrieve the artist ID based on the band name
#         if artist_id is not None:
#             # Update the user_artist_matrix with the new entry
#             user_artist_matrix[user_id, artist_id] = 1  # Set the weight as 1


def get_artist_id(artist_name):
    artist_id = None
    for artist_id, name in artist_retriever.artists.items():
        if name == artist_name:
            return artist_id
    return artist_id


if __name__ == "__main__":
    main()
