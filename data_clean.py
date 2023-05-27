import pandas as pd
from IPython.core.display_functions import display

# først skal de tre datasets loades ind med pandas ...

albums_data = pd.read_csv("datasets/spotify_albums.csv")
artists_data = pd.read_csv("datasets/spotify_artists.csv")
tracks_data = pd.read_csv("datasets/spotify_tracks.csv")

display(albums_data.head())


# Artist, genre information og album data kan joines nu.
# drop irrelevante kollonner
# kun data efter år 1990 for at holde vores dataframes små

def join_genre_and_date(artist_df, album_df, track_df):
    album = album_df.rename(columns=
                            {'id': "album_id"}).set_index('album_id')
    artist = artist_df.rename(columns=
                              {'id': "artists_id", 'name': "artists_name"}).set_index('artists_id')
    track = track_df.set_index('album_id').join(album['release_date'],
                                                on='album_id')
    track.artists_id = track.artists_id.apply(lambda x: x[2:-2])
    track = track.set_index('artists_id').join(artist[['artists_name', 'genres']],
                                               on='artists_id')
    track.reset_index(drop=False, inplace=True)
    track['release_year'] = pd.to_datetime(track.release_date, format='%Y-%m-%d').dt.year
    track.drop(
        columns=['Unnamed: 0', 'country', 'track_name_prev', 'track_number', 'type', 'lyrics', 'disc_number', 'href',
                 'instrumentalness', 'liveness', 'loudness', 'mode', 'speechiness', 'valence'],
        inplace=True)

    return track[track.release_year >= 1990]


# Herefter gør vi størrelsen på datasettet mindre ved kun at kigge efter bestemte genrer.
def get_filtered_track_df(df, genres_to_include):
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)
    [1:-1].split(", ")])
    df_exploded = df.explode("genres")[df.explode("genres")
    ["genres"].isin(genres_to_include)]
    df_exploded.loc[df_exploded["genres"] == "korean pop", "genres"] = "k-pop"
    df_exploded_indices = list(df_exploded.index.unique())
    df = df[df.index.isin(df_exploded_indices)]
    df = df.reset_index(drop=True)
    return df


genres_to_include = genres = ['dance pop', 'electronic', 'electropop', 'hip hop', 'jazz', 'k-pop', 'latin', 'pop',
                              'pop rap', 'r&b', 'rock']
track_with_year_and_genre = join_genre_and_date(artists_data, albums_data, tracks_data)
filtered_track_df = get_filtered_track_df(track_with_year_and_genre, genres_to_include)

filtered_track_df["uri"] = filtered_track_df["uri"].str.replace("spotify:track:", "")
filtered_track_df = filtered_track_df.drop(columns=['analysis_url', 'available_markets'])

pd.set_option('display.max_columns', None)
display(filtered_track_df.head())

filtered_track_df.to_csv("new_filtered_track_df.csv", index=False)
