import streamlit as st
from helper import recommend
from posters import fetch_poster

st.set_page_config(layout="wide")
st.title("🎬 AI Movie Recommendation System")

movie = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    results = recommend(movie)
    if not results:
        st.warning("Movie not found. Try another name.")
    else:
        cols = st.columns(5)
        for idx, (title, movie_id) in enumerate(results):
            with cols[idx]:
                st.text(title)
                st.image(fetch_poster(movie_id))
