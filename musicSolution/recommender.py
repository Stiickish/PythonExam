"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""


from pathlib import Path
from typing import Tuple, List

import implicit
import scipy
import matplotlib.pyplot as plt

from data import ArtistRetriever, load_user_artists
""" from recommender import data
from data import load_user_artist """

class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library with the collaborative filtering module this has.
    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
    """

    def __init__( #konstruktøren som får artist_retrieveren og implicit_modellen ind
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever #Her assignes de til en public attribute, så der kan arbejdes med dem.
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix and train the model."""
        self.implicit_model.fit(user_artists_matrix)

    def recommend( #skal have tre argumenter ind, user_id, user_artist_matrix og en int (antallet af artists vi vil have tilbage til brugeren)
        self,
        user_id: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]: #der returneres en tuple med en list af strings og en list af float (hhv en liste af artist-navne og "scores"
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[n], N=n
        )
        artists = [ #Her laves der om fra en liste af artists_id til deres navne
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores


if __name__ == "__main__":

    # load user artists matrix
    user_artists = load_user_artists(Path("../musicSolution/lastfmdata/user_artists.csv"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../musicSolution/lastfmdata/artists.dat"))

    # instantiate ALS using implicit
    implict_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01 #factors
    )

    # instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implict_model)
    recommender.fit(user_artists)
    artists, scores = recommender.recommend(2, user_artists, n=5)

    # print results
    for artist, score in zip(artists, scores):
        plt.bar(artists, scores)
        plt.xlabel("Artist")
        plt.ylabel("Score")
        plt.title("Top 5 recommendations")
        plt.xticks(rotation=45)

        # Display the plot
        plt.show()
        print(f"{artist}: {score}")



