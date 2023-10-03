import pandas as pd

from utils import convert_int
from recommendations.content_based import get_sim


def load_ids(df):
    id_map = pd.read_csv('data/links.csv')[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)

    id_map.columns = ['movieId', 'id']

    id_map = id_map.merge(df[['original_title', 'id']], on='id').set_index('original_title')

    id_map['id'] = id_map['id'].apply(convert_int)

    indices_map = id_map.set_index('id')

    return indices_map


def hybrid(df,svd,sentence_embeddings,userId, movie_name="Toy Story"):

    indices_map = load_ids(df)

    sim_scores = get_sim(df,sentence_embeddings,movie_name)

    movie_indices = [i[0] for i in sim_scores]

    movies = df.iloc[movie_indices][['original_title','id', 'vote_count', 'vote_average']]
        
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)

    movies = movies.sort_values('est', ascending=False)

    return movies.head(10)