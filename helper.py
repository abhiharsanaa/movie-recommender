import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def get_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(get_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return []
    
    movie_index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[movie_index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movies_list:
        movie_id = new_df.iloc[i[0]].movie_id
        title = new_df.iloc[i[0]].title
        recommendations.append((title, movie_id))
    return recommendations
