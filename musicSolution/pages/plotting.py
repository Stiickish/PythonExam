import pandas as pd
import matplotlib.pyplot as plt
import re

# Read data from CSV
data = pd.read_csv('../../new_filtered_track_df.csv')

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
plt.show()
