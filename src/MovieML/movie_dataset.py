from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np
import os

from movie_globals import *

class MovieDataset(Dataset):
    def __init__(self):

        users_csv = os.path.join(DATA_DIR, 'users.dat')
        #ratings_csv = os.path.join(DATA_DIR, 'ratings.dat')
        ratings_csv = os.path.join(DATA_DIR, 'ratings_1m.csv')
        movies_csv = os.path.join(DATA_DIR, 'movies.dat')
        movieslinks_csv = os.path.join(DATA_DIR, 'movies_1m.csv')

        users_headings = ['userID','gender','age','occupation','zipcode']
        ratings_headings = ['userID','movieID','rating','timestamp']
        movies_headings = ['movieID','title','genres']

        # load users
        self.users = pd.read_csv(users_csv, sep='::', engine='python', encoding='latin-1', header=None, names=users_headings)
        self.users.set_index('userID', inplace=True)
        self.users['gender'] = (self.users['gender']=='F').astype(int)
        self.users['age'] = self.users['age'].map(AGE_MAP)

        # load ratings
        #self.ratings = pd.read_csv(ratings_csv, sep='::', engine='python', encoding='latin-1', header=None, names=ratings_headings)
        self.ratings = pd.read_csv(ratings_csv, engine='python', encoding='latin-1')

        # normalize ratings
        self.ratings['rating'] = self.ratings['rating'].div(5.0)

        # load movies
        self.movies = pd.read_csv(movies_csv, sep='::', engine='python', encoding='latin-1', header=None, names=movies_headings)
        #self.movies.set_index('movieID', inplace=True)
        # encode genres with all unique possibilites
        self.movies['genres'] = pd.factorize(self.movies['genres'])[0]
        #self.movies['genres']= self.movies['genres'].str.split('|', expand = False)
        #mlb = MultiLabelBinarizer(sparse_output=True)
        # self.movies = self.movies.join(pd.DataFrame.sparse.from_spmatrix(
        #                 mlb.fit_transform(self.movies.pop('genres')),
        #                 index=self.movies.index,
        #                 columns=mlb.classes_))

        self.max_userid = self.ratings['userID'].drop_duplicates().max()
        self.max_movieid = self.ratings['movieID'].drop_duplicates().max()

        self.movielinks = pd.read_csv(movieslinks_csv, engine='python', encoding='latin-1')
        #self.movielinks.set_index('movieID', inplace=True)

        self.num_users = self.max_userid
        self.num_genders = self.users['gender'].nunique()
        self.num_ages = self.users['age'].drop_duplicates().max()
        self.num_occupations = self.users['occupation'].drop_duplicates().max()

        self.num_movies = self.max_movieid
        self.num_genres = self.movies['genres'].nunique()


    def __len__(self):
        return self.ratings.shape[0]
    
    def __getitem__(self, idx):
        rating = self.ratings.iloc[idx+1]
        user_id = rating['userID']
        #movie_id = rating['movieID']
        movie_id = rating['rowID']

        # get the rating
        rating = rating['rating']
        # get user features: gender, age, occupation
        #user = torch.from_numpy(self.users.loc[user_id][:-1].to_numpy().astype(np.float32))
        gender = self.users.loc[user_id]['gender']
        age = self.users.loc[user_id]['age']
        occupation = self.users.loc[user_id]['occupation']
        # get movie genre
        #movie = torch.from_numpy(self.movies.loc[movie_id]['genres'].to_numpy().astype(np.float32))
        # mov_idx = self.movies.loc[self.movies['movieID']==movie_id].index
        # movie_df = self.movies.loc[mov_idx]

        # movie = movie_df['title']
        # genre = movie_df['genres'].to_numpy()
        movie = self.movies.loc[movie_id]['title']
        genre = self.movies.loc[movie_id]['genres']

        # get link to movie - index and movieID do not match
        # link_idx = self.movielinks.loc[self.movielinks['movieID']==movie_id].index
        # movie_link = self.movielinks.loc[link_idx]['imdbId'].to_numpy()
        movie_link = self.movielinks.loc[movie_id]['imdbId']

        return {'user_id': int(user_id),
                'movie_id': int(movie_id),
                'gender': int(gender),
                'age': int(age),
                'occupation': int(occupation),
                #'movie': movie, 
                'genre': int(genre),
                #'link': torch.tensor(movie_link), 
                'rating': float(rating)}


if __name__ == '__main__':
    dataset = MovieDataset()

    rand_idx_array = np.random.randint(0, len(dataset), size=10)

    for idx in rand_idx_array:
        print(dataset[idx])
