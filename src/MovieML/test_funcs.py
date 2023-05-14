import pandas as pd
import numpy as np

from .movie_utils import make_epoch_chart

def test_make_epoch_chart():
    count = 50
    df = pd.DataFrame({'a': np.random.rand(count), 'b': np.random.rand(count)})
    make_epoch_chart(df, 'Random Numbers', 'Random', 'random_numbers', show=True)


if __name__ == '__main__':
    test_make_epoch_chart()