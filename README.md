# Movie Recommender System

## Introduction

This project is a content-based movie recommender system built using the MovieLens dataset and TensorFlow’s Functional API. The system leverages deep neural networks to learn user and movie representations for personalized recommendations. By modeling content features and user preferences, the system generates recommendations tailored to each user's interests. This project is an excellent demonstration of end-to-end machine learning workflow—from data preprocessing and model training to real-time inference with an interactive Streamlit UI.

## Data Preprocessing

The data preprocessing pipeline involves several key steps to convert raw MovieLens data into feature sets for both movies and users:

- **Movie Data Processing:**
  - **Filtering by Release Year:**  
    Only movies released after the year 2000 are considered to ensure we work with recent content.
  - **Genre Filtering:**  
    The genres for each movie are split into a list. Only movies whose genres are a subset of a predefined list of popular genres (e.g., *Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller*) are kept.
  - **One-Hot Encoding:**  
    For each popular genre, a binary indicator is created. For instance, if a movie's genre list includes "Action", a corresponding column is set to 1.
  - **Average Rating Calculation:**  
    The average rating of each movie is computed from user ratings and merged into the movie features.
    
- **User Data Processing:**
  - **Exploding Genres:**  
    The movies DataFrame is exploded so that each movie appears once per genre. This allows us to associate each user’s rating with a specific genre.
  - **Per-Genre Average Ratings:**  
    For each user, the average rating per genre is computed using group-by and unstacking. The result is a user profile where each row represents a user, and each column (apart from `userId`) represents the user’s average rating for that genre.
  - **Merging Data:**  
    The user ratings (from the raw ratings file) are merged with these computed user profiles and with the movie features to create a comprehensive training set that captures user–movie interactions.

This preprocessing pipeline transforms raw data into structured numerical features that feed into the deep learning model.

## Model Architecture

The model is built using **two separate networks**:

- **User Network:**  
  Processes user feature vectors derived from per-genre average ratings.  
  **Architecture:**  
  - Input layer → Dense (256, ReLU) → Dense (128, ReLU) → Dense (32)  
  - **Normalization:** An L2 normalization layer (implemented via a Lambda layer) is applied to output unit vectors.

- **Movie Network:**  
  Processes movie feature vectors including release year, one-hot encoded genre features, and average ratings.  
  **Architecture:**  
  - Input layer → Dense (256, ReLU) → Dense (128, ReLU) → Dense (32)  
  - **Normalization:** Similarly, an L2 normalization layer is applied to generate unit-normalized movie embeddings.

After obtaining normalized embeddings, the **cosine similarity** between the user and movie vectors is computed as a simple dot product. This similarity score is used to rank movies for recommendation.

## Training and Evaluation

- **Loss Curve:**  
  The training process is monitored using a loss curve, which is saved as an image (e.g., `training_loss_curve.png`).  
  *[Insert training loss, test loss details, and the loss curve image here]*

- **Scalers:**  
  Feature inputs and target ratings are normalized using `StandardScaler` and `MinMaxScaler`, respectively.

## Cold Start Solution

- **New Users:**  
  New users are prompted to rate their interest in various movie genres via interactive sliders (with a range from 0 to 5 in 0.5 increments). These ratings directly form the user's profile.
  
- **Existing Users:**  
  For returning users, the system updates their profile using an **exponential weighted average** with a beta value of **0.8**. This approach balances historical ratings with new inputs to adapt recommendations as user preferences evolve.

## Demo Video

Watch this demo video of the working system that demonstrates how the Streamlit app recommends movies based on user inputs:

[![Movie Recommender Demo](path/to/demo_thumbnail.png)](https://youtu.be/your_video_link)

*(Replace the placeholder paths and links with your actual video thumbnail and URL.)*

## Future Enhancements

- **Hybrid Recommendations:**  
  Explore integrating collaborative filtering methods to complement content-based predictions.
- **Enhanced Feature Engineering:**  
  Incorporate additional metadata (e.g., cast, director, movie synopsis) to further enrich movie representations.
- **UI Improvements:**  
  Add movie posters, detailed descriptions, and interactive visualizations in the Streamlit app.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


