import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
pd.set_option("display.precision", 1)
import keras
import re
import streamlit as st

movies=pd.read_csv('movie-lens/ml-latest-small/movies.csv')
ratings=pd.read_csv('movie-lens/ml-latest-small/ratings.csv')
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
filt=movies['year'] > 2000
movies = movies[filt]
popular_genres=['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Horror','Mystery','Romance','Sci-Fi','Thriller']
movies['genres'] = movies['genres'].str.split('|')
movies = movies[movies['genres'].apply(lambda g: set(g).issubset(popular_genres))]
genres = sorted(set(g for sublist in movies['genres'] for g in sublist))
for genre in genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)
filtered_ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
movie_avg_ratings = filtered_ratings.groupby('movieId')['rating'].mean().reset_index()
movies = movies.merge(movie_avg_ratings, on='movieId', how='left')
movies.rename(columns={'rating':'avg_rating'},inplace=True)
movies_exploded = movies.explode('genres')[['movieId', 'genres']]
ratings_with_genres = filtered_ratings.merge(movies_exploded, on='movieId', how='left')
user_genre_avg = ratings_with_genres.groupby(['userId', 'genres'])['rating'].mean().unstack(fill_value=0).reset_index()
training_set = filtered_ratings.merge(user_genre_avg, on='userId', how='left')

# Merge the result with movie features (by movieId)
training_set = training_set.merge(movies, on='movieId', how='left')
num_repeats = 2  # Increase the dataset 5-fold
training_set = pd.concat([training_set] * num_repeats, ignore_index=True)

# Extract the output label y (the rating)
y = training_set['rating'].values
# Extract user features
user_feature_cols =[col for col in training_set.columns if col.endswith('_x')]
user_train = training_set[user_feature_cols].to_numpy()
# Extract movie features
movie_feature_cols =['year']+ [col for col in training_set.columns if col.endswith('_y')]+['avg_rating']
movie_train = training_set[movie_feature_cols].to_numpy()

# normalising the trainig set
scaler_movie=StandardScaler()
scaler_movie.fit(movie_train)
movie_train_scaled=scaler_movie.transform(movie_train)
scaler_user=StandardScaler()
scaler_user.fit(user_train)
user_train_scaled=scaler_user.transform(user_train)
scaler_target=MinMaxScaler((-1,1))
scaler_target.fit(y.reshape(-1,1))
y_scaled=scaler_target.transform(y.reshape(-1,1))
movie_train, movie_test = train_test_split(movie_train_scaled, train_size=0.90, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train_scaled, train_size=0.90, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_scaled,    train_size=0.90, shuffle=True, random_state=1)
keras.config.enable_unsafe_deserialization()

num_outputs=32
@tf.keras.utils.register_keras_serializable()
def my_l2_normalize(x):
    return tf.linalg.l2_normalize(x, axis=1)
# Saved Model
model = tf.keras.models.load_model("my_model.keras",custom_objects={'my_l2_normalize': my_l2_normalize})

def update_user_genre_ratings(user_id, new_ratings, user_df, alpha=0.8):
    if user_id in user_df['userId'].values:
        idx = user_df.index[user_df['userId'] == user_id][0]
        for genre, new_rating in new_ratings.items():
            existing_rating = user_df.at[idx, genre]
            updated_rating = (1 - alpha) * existing_rating + alpha * new_rating
            user_df.at[idx, genre] = updated_rating
        st.write(f"User {user_id} updated with new ratings.")
    else:
        new_entry = {'userId': user_id}
        for genre in user_df.columns:
            if genre != 'userId':
                new_entry[genre] = new_ratings.get(genre, 0)  # Default to 0 if not provided.
        user_df = pd.concat([user_df, pd.DataFrame([new_entry])], ignore_index=True)
        st.write(f"New user {user_id} added.")
    return user_df
movie_vecs=movies.to_numpy()
movie_vecs=movies.to_numpy()
def get_recommendations(user_id,user_df):
    filt=user_df['userId'] == user_id
    row = user_df.loc[filt].iloc[0].drop('userId').to_numpy()
    repeat_rows=np.tile(row,(movie_vecs.shape[0],1))
    suser_vecs = scaler_user.transform(repeat_rows)
    smovie_vecs = scaler_movie.transform(movie_vecs[:,3:])
    y_p=model.predict([suser_vecs,smovie_vecs])
    # unscale y prediction 
    y_pu = scaler_target.inverse_transform(y_p)
    # sort the results, highest prediction first
    sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
    sorted_ypu   = y_pu[sorted_index]
    sorted_movies = pd.DataFrame(movie_vecs[sorted_index]) #using unscaled vectors for display
    sorted_movies['Pred_rating']=sorted_ypu
    sorted_movies.rename(columns={1:'Title',2:'Genre'},inplace=True)
    output_df=sorted_movies[['Title','Genre','Pred_rating']]
    return output_df.head(10)
st.title("Movie Recommender System")

# User inputs their ID and genre ratings.
user_id = st.number_input("Enter your User ID (if new, enter a new ID):", min_value=1, step=1)
genres = popular_genres

new_ratings = {}
for genre in genres:
    new_ratings[genre] = st.slider(f"Rating for {genre}", min_value=0, max_value=5, value=0)

if st.button("Submit Ratings"):
    # Update (or add) the user in the user_genre_avg DataFrame.
    user_genre_avg = update_user_genre_ratings(user_id, new_ratings, user_genre_avg, alpha=0.8)
    
    # Now generate recommendations using the updated user_genre_avg.
    rec_df = get_recommendations(user_id, user_genre_avg)
    
    st.write("### Top Movie Recommendations:")
    st.dataframe(rec_df)
