import pandas as pd

from pathlib import Path
import os

MOD_DUR = os.path.dirname(Path(__file__).absolute())
PROJ_PATH = os.path.join(MOD_DUR, '..', '..')

M1_DATASET = 'ml-1m'
DATA_DIR = os.path.join(PROJ_PATH, 'data', M1_DATASET)

movies_1m = pd.read_csv(os.path.join(DATA_DIR, 'movies.dat'), sep='::', header=None, encoding='latin-1', names=['movieId', 'title', 'genres'])
#movies_1m['genres']= movies_1m['genres'].str.split('|', expand = False)
movie_id = 1123

movie_df = movies_1m[movies_1m['movieId'] == movie_id]

print(movie_df)
