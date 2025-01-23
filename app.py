import streamlit as st
import pickle
import pandas as pd

# Function to recommend movies
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    
    for i in distances[1:6]:
        # Fetch the movie title
        recommended_movie_names.append(movies.iloc[i[0]].title)
    
    return recommended_movie_names  # Explicitly return the list

# Load data
movies_dict = pickle.load(open('movies_dict', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Streamlit app
st.title('Movie Recommender System')

# Movie selection dropdown
selected_movie = st.selectbox(
    "Select a movie:",
    movies['title'].values
)

# Recommendation button
if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    
    # Display recommendations
    for movie in recommendations:
        st.write(movie)
