from sklearn.neighbors import NearestNeighbors


# def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
#     genre = genre.lower()
#     genre_data = exploded_track_df[
#         (exploded_track_df["genres"] == genre) & (exploded_track_df["release_year"] >= start_year) & (
#                     exploded_track_df["release_year"] <= end_year)]
#     genre_data = genre_data.sort_values(by='popularity', ascending=False)[:1000]
#     neigh = NearestNeighbors()
#     neigh.fit(genre_data[audio_feats].to_numpy())
#     n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
#     uris = genre_data.iloc[n_neighbors]["uri"].tolist()
#     audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
#     return uris, audios

"""
1. We convert the genre parameter to lowercase, to ensure consistent formatting
2. We filter exploded_track_df based on 3 conditions
    a. The genres column is equal to the specified genre
    b. The release_year column is greater or equal to start_year
    c. The release_year column is less or equal to end_year
3. We then sort the list by popularity and limits it to the first 1000 rows using a slicing syntax
4. We assign a instance of NearestNeighbors clas to our variable neigh
5. The fit method of the neigh object is called with the input data as genre_data[audio_feats].to_numpy(). 
    This line assumes that audio_feats is a list of column names representing audio features in the genre_data DataFrame
6. The kneighbors method of the neigh object is called with the test_feat parameter as input. 
    It returns the indices of the nearest neighbors in the fitted data. The n_neighbors variable is assigned the result for the first element [0] of the returned indices.
7. The uris variable is assigned the "uri" values from the genre_data DataFrame, corresponding to the indices stored in n_neighbors. These values are converted to a Python list using the tolist() method.
8. The audios variable is assigned the audio feature values from the genre_data DataFrame, corresponding to the indices stored in n_neighbors. These values are extracted as a NumPy array using the to_numpy() method.
9. Return uris and audios as tuples
"""