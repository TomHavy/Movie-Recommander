import pandas as pd
import sys

sys.path.append("C:/Users/thavyarimana/RecSys/src")

from recommendations.content_based import (
    get_similar_movies,
    load_embeddings,
)


df = pd.read_csv('data/cleaned.csv')

sentence_embeddings = load_embeddings('embeddings.pickle')

reco =  get_similar_movies(df,sentence_embeddings,movie_name="Toy Story")

print(reco)