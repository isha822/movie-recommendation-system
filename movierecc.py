import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pickle


movies = pd.read_csv(r'C:\Users\user\Desktop\tmdb_5000_movies.csv')
credits = pd.read_csv(r'C:\Users\user\Desktop\tmdb_5000_credits.csv')
movies = movies.merge(credits,on='title')
#we are importing movie and credits data and then merging them together
#now we will remove a few criteria from our data which is not necessary for movie reccamendation criteria
# we will keep genres id keywords title overview cast crew
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.isnull().sum() #displays missing data
movies.dropna(inplace=True) #drops the issing data
movies.duplicated().sum() #removes duplication
def convert(obj): #function to convrt grnres ihnto a list of data 
    L = []
    for i in ast.literal_eval(obj): #this library fun does it
         L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert) #then we applying convert fun to these data 
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj): # similar function to convert cast
     L =[]
     counter = 0
     for i in ast.literal_eval(obj):
          if counter != 3:
             L.append(i['name'])
             counter +=1
          else:
              break
     return L   

movies['cast']=  movies['cast'].apply(convert3) 

def fetch_director(obj): #this fun is to find the directors name from our data
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director) #using fetch fun 
movies['overview'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') #splitting string into list os words
movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x)) 
movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join(x))
movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))
movies['crew'] = movies['crew'].apply(lambda x: ' '.join(x))
movies['tags'] = movies['overview'] + movies['cast'] + movies['crew'] + movies['keywords'] + movies['genres']
new_df = movies[['movie_id', 'title', 'tags']].copy()#creating a new data frame which we will be using 
new_df['tags'] =  new_df['tags'].apply(lambda x:x.lower()) #for joining two words as one "joins its elements into a single string"
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) #to lowercase the words which is assumed to be a string 

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)



new_df['tags'] = new_df['tags'].apply(stem) 
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
# this is to convert the text into a matrix of tokrn counts
cv.get_feature_names_out()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

#sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[1]].title)
        
# ...existing code...

def recommend(movie):
    # Normalize input for matching
    movie = movie.lower().strip()
    # Find the index in a case-insensitive way
    matches = new_df[new_df['title'].str.lower().str.strip() == movie]
    if matches.empty:
        print("Movie not found. Please check the spelling or try another movie.")
        return
    movie_index = matches.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

def ask_and_recommend(): 
    while True:
        movie = input("Enter a movie name: ")
        recommend(movie)
        more = input("Do you want to see more recommendations? (yes/no): ")
        if more.lower() != 'yes':
            print("Thank you for using our movie recommendation system.")
            break

ask_and_recommend()


pickle.dump(new_df, open(r'C:\Users\user\Desktop\movies.pkl', 'wb'))

pickle.dump(similarity, open(r'C:\Users\user\Desktop\similarity.pkl', 'wb'))

import os
print(os.path.exists(r'C:\Users\user\Desktop\similarity.pkl'))