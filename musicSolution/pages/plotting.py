import plotly.express as px
import streamlit as st
import pandas as pd
import os
import ast

# Get the absolute path of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path relative to the script directory
file_path = os.path.join(script_dir, 'new_filtered_track_df.csv')

df = pd.read_csv(file_path)

# Convert the genres column from string to list
df['genres'] = df['genres'].apply(ast.literal_eval)

# Create a list of unique genres for the dropdown
genres_list = df['genres'].explode().unique()

# Add a dropdown menu for genre selection
selected_genre = st.selectbox("Select Genre", genres_list)

# Filter the DataFrame based on the selected genre
df_selected_genre = df[df['genres'].apply(lambda x: selected_genre in x)]

# Calculate the average popularity for each release year
df_avg_popularity = df_selected_genre.groupby('release_year')['popularity'].mean().reset_index()

fig = px.line(df_avg_popularity, x="release_year", y="popularity")

fig.update_layout(
    title=f"Average Popularity of {selected_genre.capitalize()} Genre Over the Years",
    xaxis_title="Release Year",
    yaxis_title="Average Popularity"
)

st.plotly_chart(fig)
