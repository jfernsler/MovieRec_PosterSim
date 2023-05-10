import pandas as pd

from pathlib import Path
import os

MOD_DUR = os.path.dirname(Path(__file__).absolute())
PROJ_PATH = os.path.join(MOD_DUR, '..', '..')

M1_DATASET = 'ml-1m'
DATA_DIR = os.path.join(PROJ_PATH, 'data', M1_DATASET)

movies_25m = pd.read_csv(os.path.join(DATA_DIR, 'movies_25m.csv'))

links_25m = pd.read_csv(os.path.join(DATA_DIR, 'links_25m.csv'))

movies_1m = pd.read_csv(os.path.join(DATA_DIR, 'movies.dat'), sep='::', header=None, encoding='latin-1', names=['movieId', 'title', 'genres'])

merge_25m = pd.merge(links_25m, movies_25m, on='movieId')
merge_25m = merge_25m[['imdbId', 'title']]
#merge_25m.to_csv(os.path.join(DATA_DIR, 'movieLink.csv'), index=False)

print(merge_25m.head())
print(merge_25m.shape)

merge_movies = pd.merge(movies_1m, merge_25m, on='title', how='left')
#merge_movies.to_csv(os.path.join(DATA_DIR, 'movies_1m.csv'), index=False)
print(merge_movies[merge_movies['imdbId'].isna()].head())
