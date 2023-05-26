from pathlib import Path
from typing import Tuple, List
import implicit
import scipy
from sklearn.neighbors import NearestNeighbors
from musicSolution.data import load_user_artists, ArtistRetriever


class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
    """

    def __init__(
            self,
            artist_retriever: ArtistRetriever,
            implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(user_artists_matrix)

    def recommend(
            self,
            user_id: int,
            user_artists_matrix: scipy.sparse.csr_matrix,
            n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores


class KNNRecommender:
    """The KNNRecommender class computes recommendations for a given user
    using the implicit library with k-nearest neighbors.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - knn_model: a NearestNeighbors model
    """

    def __init__(
            self,
            artist_retriever: ArtistRetriever,
            knn_model: NearestNeighbors,
    ):
        self.artist_retriever = artist_retriever
        self.knn_model = knn_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix."""
        self.knn_model.fit(user_artists_matrix)

    def recommend(
            self,
            user_id: int,
            user_artists_matrix: scipy.sparse.csr_matrix,
            n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        user_vector = user_artists_matrix[user_id].toarray()
        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=n + 1)
        # Exclude the first index, which represents the user itself
        indices = indices.flatten()[1:]
        distances = distances.flatten()[1:]

        artist_ids = [user_artists_matrix.indices[i] for i in indices]
        scores = [1 - d for d in distances]

        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores


if __name__ == "__main__":
    # Load user artists matrix
    user_artists = load_user_artists(Path("../datasets/user_artists.dat"))

    # Instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../datasets/artists.dat"))

    # Instantiate KNN model using scikit-learn
    knn_model = NearestNeighbors(n_neighbors=200, algorithm='brute', metric='cosine')

    # Fit the KNN model with user artists matrix
    knn_model.fit(user_artists)

    # Instantiate recommender, fit, and recommend
    recommender = KNNRecommender(artist_retriever, knn_model)
    recommender.fit(user_artists)
    artists, scores = recommender.recommend(2, user_artists, n=20)

    # Print results
    for artist, score in zip(artists, scores):
        print(f"This is the results for the KNN-algorithm: {artist}: {score}")
