imprt numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
pd.set_option("display.precision", 1)
import keras
import re

# importing the ml-latest-small dataaset
movies=pd.read_csv('movie-lens/ml-latest-small/movies.csv')
ratings=pd.read_csv('movie-lens/ml-latest-small/ratings.csv')

# PREPARING THE MOVIES TRAINING SET
# only taking movies of popular genres and movies after year 2000
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
filt=movies['year'] > 2000
movies = movies[filt]
popular_genres=['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Horror','Mystery','Romance','Sci-Fi','Thriller']
movies['genres'] = movies['genres'].str.split('|')
movies = movies[movies['genres'].apply(lambda g: set(g).issubset(popular_genres))]
genres = sorted(set(g for sublist in movies['genres'] for g in sublist))
for genre in genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)

# getting ratings for the selected movies only
filtered_ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
movie_avg_ratings = filtered_ratings.groupby('movieId')['rating'].mean().reset_index()
movies = movies.merge(movie_avg_ratings, on='movieId', how='left')
movies.rename(columns={'rating':'avg_rating'},inplace=True)

# PREPARING USER TRAINING SET
movies_exploded = movies.explode('genres')[['movieId', 'genres']]
ratings_with_genres = filtered_ratings.merge(movies_exploded, on='movieId', how='left')

# Now group by userId and genre_list and calculating the average rating 
user_genre_avg = ratings_with_genres.groupby(['userId', 'genres'])['rating'].mean().unstack(fill_value=0).reset_index()

# MERGING THE MOVIE AND USER TRAINING SET TO GET THE USER AND MOVIE INTERACTION

training_set = filtered_ratings.merge(user_genre_avg, on='userId', how='left')
training_set = training_set.merge(movies, on='movieId', how='left')

num_repeats = 2  # Increase the dataset 2-fold
training_set = pd.concat([training_set] * num_repeats, ignore_index=True)
# ExtractING the output label y 
y = training_set['rating'].values

# GETTING FINAL USER AND MOVIE TRAINING ARRAY WITH REQUIRED FEATURES
# Extracting user features
user_feature_cols =[col for col in training_set.columns if col.endswith('_x')]
user_train = training_set[user_feature_cols].to_numpy()
# Extract movie features
movie_feature_cols =['year']+ [col for col in training_set.columns if col.endswith('_y')]+['avg_rating']
movie_train = training_set[movie_feature_cols].to_numpy()

# NORMALISING THE MOVIE AND USER TRAINING SET
scaler_movie=StandardScaler()
scaler_movie.fit(movie_train)
movie_train_scaled=scaler_movie.transform(movie_train)
scaler_user=StandardScaler()
scaler_user.fit(user_train)
user_train_scaled=scaler_user.transform(user_train)
scaler_target=MinMaxScaler((-1,1))
scaler_target.fit(y.reshape(-1,1))
y_scaled=scaler_target.transform(y.reshape(-1,1))

# DIVIDING THE DATASET IN TRAIN AND TEST SET
movie_train, movie_test = train_test_split(movie_train_scaled, train_size=0.90, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train_scaled, train_size=0.90, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_scaled,    train_size=0.90, shuffle=True, random_state=1)
