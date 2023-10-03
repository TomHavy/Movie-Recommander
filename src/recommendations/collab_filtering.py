import pandas as pd

from surprise.model_selection import cross_validate
from surprise import (
    Reader, 
    Dataset, 
    SVD,
)


def load_ratings():

    reader = Reader()

    ratings = pd.read_csv('data/ratings_small.csv')
    ratings.head()

    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    return data


def model(data):
    svd = SVD()
    cross_validate(svd, data, measures=['RMSE', 'MAE'],cv=10, verbose=True)

    trainset = data.build_full_trainset()
    
    svd.fit(trainset)

    return svd


def get_rating(svd,user_id,movie_id):

    pred = svd.predict(user_id, movie_id)

    return f"Note: {round(pred.est,3)}/5"

