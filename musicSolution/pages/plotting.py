import pandas as pd
import matplotlib.pyplot as plt
import re
import plotly.express as px
import streamlit as st
import os
import ast

# Get the absolute path of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path relative to the script directory
file_path = os.path.join(script_dir, 'new_filtered_track_df.csv')

df = pd.read_csv(file_path)
data = pd.read_csv(file_path)

# Choose the columns for x-axis, y-axis, and release year
x_column = 'artists_name'
y_column = 'popularity'
year_column = 'release_year'

# Handle special character in artist name
data[x_column] = data[x_column].apply(lambda name: re.sub(r'\$\$', '$', name))

# Sort the data based on popularity column and select top 20 rows
top_20_popular = data.sort_values(by=y_column, ascending=False).head(20)

# Extract the data for the selected columns
artists = top_20_popular[x_column]
popularity = top_20_popular[y_column]
years = top_20_popular[year_column]

# Combine artist names and release years for x-axis labels
labels = [f'{artist} ({year})' for artist, year in zip(artists, years)]

# Plotting the bar chart
plt.bar(labels, popularity)
plt.xlabel('Artists (Release Year)')
plt.ylabel(y_column)
plt.title(f'Top 20 {y_column} of {x_column} by Release Year')
plt.xticks(rotation=90)
plt.tight_layout()

show_popular_artists = st.button("Show top 20 artists")
if show_popular_artists:
    st.pyplot(plt)

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
