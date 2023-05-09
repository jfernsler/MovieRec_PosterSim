from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import os

from movie_globals import DATA_DIR

class MovieDataset(Dataset):
    def __init__(self):

        users_csv = os.path.join(DATA_DIR, 'users.dat')
        ratings_csv = os.path.join(DATA_DIR, 'ratings.dat')
        movies_csv = os.path.join(DATA_DIR, 'movies.dat')

        users_headings = ['userID','gender','age','occupation','zipcode']
        ratings_headings = ['userID','movieID','rating','timestamp']
        movies_headings = ['movieID','title','genres']

        # load users
        self.users = pd.read_csv(users_csv, sep='::', engine='python', encoding='latin-1', names=users_headings)

        # load ratings
        self.ratings = pd.read_csv(ratings_csv, sep='::', engine='python', encoding='latin-1', names=ratings_headings)
        # normalize ratings
        self.ratings['rating'] = self.ratings['rating'].div(5.0)

        # load movies
        self.movies = pd.read_csv(movies_csv, sep='::', engine='python', encoding='latin-1', names=movies_headings)
        # encode genres
        self.movies['genres']= self.movies['genres'].str.split('|', expand = False)
        mlb = MultiLabelBinarizer(sparse_output=True)
        self.movies = self.movies.join(pd.DataFrame.sparse.from_spmatrix(
                        mlb.fit_transform(self.movies.pop('genres')),
                        index=self.movies.index,
                        columns=mlb.classes_))



if __name__ == '__main__':
    dataset = MovieDataset()
    print(dataset.movies.head())
    print(dataset.users.head())
    print(dataset.ratings.head())