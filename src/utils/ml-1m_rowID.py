import pandas as pd

from pathlib import Path
import os

MOD_DUR = os.path.dirname(Path(__file__).absolute())
PROJ_PATH = os.path.join(MOD_DUR, '..', '..')

M1_DATASET = 'ml-1m'
DATA_DIR = os.path.join(PROJ_PATH, 'data', M1_DATASET)

movies_1m = pd.read_csv(os.path.join(DATA_DIR, 'movies.dat'), sep='::', engine='python', header=None, encoding='latin-1', names=['movieId', 'title', 'genres'])
ratings_1m = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::', engine='python', header=None, encoding='latin-1', names=['userId', 'movieId', 'rating', 'timestamp'])

rowID = []
for ind in ratings_1m.index:
    if ind%1000 == 0:
        print(ind, rowID[-10:])
    rowID.append(movies_1m.loc[movies_1m['movieId'] == ratings_1m.loc[ind]['movieId']].index[0])

ratings_1m['rowID'] = rowID
ratings_1m.to_csv(os.path.join(DATA_DIR, 'ratings_1m.csv'), index=False)

print(ratings_1m.head())