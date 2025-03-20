import numpy as np
import pandas as pd
import streamlit as st

# function for adding a new user or updating the rating for a existing user
def update_user_genre_ratings(user_id, new_ratings, user_df, alpha=0.8),scaler_user,scaler_movie,scaler_target:
    if user_id in user_df['userId'].values:
        idx = user_df.index[user_df['userId'] == user_id][0]
        for genre, new_rating in new_ratings.items():
            existing_rating = user_df.at[idx, genre]
            updated_rating = (1 - alpha) * existing_rating + alpha * new_rating
            user_df.at[idx, genre] = updated_rating
        #st.write(f"User {user_id} updated with new ratings.")
    else:
        new_entry = {'userId': user_id}
        for genre in user_df.columns:
            if genre != 'userId':
                new_entry[genre] = new_ratings.get(genre, 0)  # Default to 0 if not provided.
        user_df = pd.concat([user_df, pd.DataFrame([new_entry])], ignore_index=True)
        #st.write(f"New user {user_id} added.")
    return user_df
movie_vecs=movies.to_numpy()

# Getting recommendations for the user which is updated or added in the user_genre_avg
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