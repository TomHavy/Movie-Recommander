import pandas as pd
import sys

sys.path.append("C:/Users/thavyarimana/RecSys/src")

from recommendations.hybrid import hybrid
from recommendations.collab_filtering import (
    model,
    load_ratings
)
from recommendations.content_based import(
    load_embeddings
)


df = pd.read_csv('data/cleaned.csv')

data = load_ratings()

svd = model(data)

sentence_embeddings = load_embeddings('embeddings.pickle')

reco = hybrid(df,svd,sentence_embeddings,userId=1, movie_name="Toy Story")

print(reco)
