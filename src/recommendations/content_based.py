
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from utils import index


def load_embeddings(path):

    with open(path, 'rb') as pkl:
        sentence_embeddings = pickle.load(pkl)

    print("Loaded embeddings")
    return sentence_embeddings


def get_sim(df,sentence_embeddings,movie_name):

    similarity = cosine_similarity(sentence_embeddings)

    sim_scores = sorted(list(enumerate(similarity[index(df,movie_name)])), key = lambda x:x[1], reverse= True)
    
    sim_scores = sim_scores[1:26]
    
    return sim_scores


def get_similar_movies(df,sentence_embeddings,movie_name):

    sim_scores = get_sim(df,sentence_embeddings,movie_name)

    movie_indices = [i[0] for i in sim_scores]
    
    movies = df.iloc[movie_indices][['original_title', 'vote_count', 'vote_average']]

    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    
    m = vote_counts.quantile(0.65)
    
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]

    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified = qualified.sort_values('vote_average', ascending=False).head(10)

    return qualified