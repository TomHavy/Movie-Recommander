import pandas as pd
from ast import literal_eval


def _load():
    credits = pd.read_csv('data/credits.csv')
    keywords = pd.read_csv('data/keywords.csv')
    movies = pd.read_csv('data/movies_metadata.csv').\
                        drop(['belongs_to_collection', 'homepage', 'imdb_id', 'poster_path', 'status', 'title', 'video'], axis=1).\
                        drop([19730, 29503, 35587]) # Incorrect data type

    movies['id'] = movies['id'].astype('int64')

    df = movies.merge(keywords, on='id').\
        merge(credits, on='id')
    
    return df


def _get_text(text, obj='name'):
    text = literal_eval(text)
    
    if len(text) == 1:
        for i in text:
            return i[obj]
    else:
        s = []
        for i in text:
            s.append(i[obj])
        return ', '.join(s)


def _clean(df):

    df['original_language'] = df['original_language'].fillna('')
    df['runtime'] = df['runtime'].fillna(0)
    df['tagline'] = df['tagline'].fillna('')

    df.dropna(inplace=True)

    df['genres'] = df['genres'].apply(_get_text)
    df['production_companies'] = df['production_companies'].apply(_get_text)
    df['production_countries'] = df['production_countries'].apply(_get_text)
    df['crew'] = df['crew'].apply(_get_text)
    df['spoken_languages'] = df['spoken_languages'].apply(_get_text)
    df['keywords'] = df['keywords'].apply(_get_text)

    return df


def _combine_features(row):
    return row["original_title"] + " " + row["genres"] + " " +  row["overview"] + " " + row["keywords"] + " " \
        + row["original_language"] + " " + row["production_companies"] + " " + row["production_countries"] + " "\
        + row['crew'] + " " + row['characters'] + " " + row['actors']


def _feature_eng(df):
    # New columns
    df['characters'] = df['cast'].apply(_get_text, obj='character')
    df['actors'] = df['cast'].apply(_get_text)

    df.drop('cast', axis=1, inplace=True)
    df = df[~df['original_title'].duplicated()]
    df = df.reset_index(drop=True)

    df["combined"] = df.apply(_combine_features, axis=1)    

    return df


def get_clean_data():
    df = _load()

    df = _clean(df)

    df = _feature_eng(df)

    df.to_csv("data/cleaned.csv",index=False)

    return df