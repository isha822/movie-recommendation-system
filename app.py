import streamlit as st
import  pickle
import pandas as pd

def recommend(movie):
    #movie = movie.lower().strip()
    movie_index =  movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:8]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


similarity = pickle.load(open('similarity.pkl', 'rb'))

movies_data = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_data)
movies_list = movies['title'].values

st.title('Movie Recommender system')

#movies = (movies_list)
selected_movie = st.selectbox('enter movie name', movies_list)



if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    for i in recommendations:
         st.write(i)




