import streamlit as st
import pandas as pd
from utils.py import *

user_genre_avg=pd.read_csv('user_genre_avg.csv')  # dataframe consisting the user data
popular_genres= ['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Horror','Mystery','Romance','Sci-Fi','Thriller']

st.title("Movie Recommender System")

# User inputs their ID and genre ratings
user_id = st.number_input("Enter your User ID (if new, enter a new ID):", min_value=1, step=1)

genres = popular_genres

new_ratings = {}
for genre in genres:
    new_ratings[genre] = st.slider(f"Rating for {genre}", min_value=0, max_value=5, value=3,step=0.5)

if st.button("Submit Ratings"):
    # Update the new user in the datframe or average the ratings for a existing user
    user_genre_avg = update_user_genre_ratings(user_id, new_ratings, user_genre_avg, alpha=0.8,scaler_user,scaler_movie,scaler_target)
    
    # Now generate recommendations using the updated user_genre_avg 
    rec_df = get_recommendations(user_id, user_genre_avg)
    
    st.write("### Top Movie Recommendations:")
    st.dataframe(rec_df)