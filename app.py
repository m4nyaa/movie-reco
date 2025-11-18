import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("moviebh.csv")

movies["combined"] = movies["title"] + " " + movies["genres"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["combined"])

similarity = cosine_similarity(tfidf_matrix)

def recommend(movie_name):
    movie_name = movie_name.lower().strip()

    for idx, title in enumerate(movies["title"]):
        if title.lower() == movie_name:
            movie_index = idx
            break
    else:
        return []

    scores = list(enumerate(similarity[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = [movies.iloc[i]["title"] for i, _ in sorted_scores]
    return recommended_movies


st.title("🎬 Movie Recommendation System")
st.subheader("Bollywood + Hollywood Trending Mix")

st.write("Select a movie from the list below and get recommendations!")

selected_movie = st.selectbox("Choose a movie:", movies["title"].values)

if st.button("Get Recommendations"):
    results = recommend(selected_movie.lower())
    
    st.write("### ⭐ Recommended Movies:")
    for movie in results:
        st.write("- ", movie)
