import pandas as pd
import seaborn  as sns
import matplotlib.pyplot as plt
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pickle

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

print(movies.head())
print(credits.head())
movies=movies.merge(credits,on='title')
print(movies.shape)
print(movies.info())

'''
genres
id
keywords
title
overview
'''
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.shape)
print(movies.head())
print(movies.isnull().sum())
movies.dropna(inplace=True)
print(movies.duplicated().sum())
print(movies['genres'])
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)


def convert3(obj):
 l=[]
 counter=0
 for i in ast.literal_eval(obj):
    if counter!=3:
        l.append(i['name'])
        counter+=1
    else:
        break
 return l

movies['cast']=movies['cast'].apply(convert3)

def fetch_director(obj):
     l = []
     for i in ast.literal_eval(obj):
         if i['job']=='Director':
          l.append(i['name'])
          break
     return l
movies['crew']=movies['crew'].apply(fetch_director)

movies['overview']=movies['overview'].apply(lambda x:x.split())
print(movies.head())

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags']=movies['overview']+movies["genres"]+movies["keywords"]+movies["cast"]

new_df=movies[['movie_id','title','tags']].copy()
print(new_df)

new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
print(vectors)

#steming ['love','loving']=love

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

# we will calculate cosine distance, we will find angle between these.
# vector graph distance inverse propostional to similarity

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
print("cosine distance score ")
print(similarity)

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    return

recommend("Batman Begins")

pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))








































