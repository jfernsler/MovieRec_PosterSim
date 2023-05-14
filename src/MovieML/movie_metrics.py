import torch

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from .movie_dataset import MovieDataset
from .movie_globals import *
from .movie_poster_sim import *
from .movie_predict import get_model

def matrix_all():
    # evaluate the model
    movie_data = MovieDataset()
    num_movies = 1000
    movie_loader = torch.utils.data.DataLoader(movie_data,
                                                batch_size=num_movies,
                                                shuffle=False,
                                                num_workers=8)

    device = torch.device('cpu')
    model = get_model(movie_data)
    model.load_state_dict(torch.load(MODEL_STATE))
    model.eval()

    y_list = list()
    yhat_list = list()

    for n,d in enumerate(movie_loader):
        print(f'{n} ', end='', flush=True)
        # X
        user_ids_batch = d['user_id'].to(device)
        genders_batch = d['gender'].to(device)
        ages_batch = d['age'].to(device)
        occupations_batch = d['occupation'].to(device)
        movie_ids_batch = d['movie_id'].to(device)
        genres_batch = d['genre'].to(device)
        # Y
        y =  d['rating'].to(device)
        # Predict
        yhat = model(user_ids_batch, genders_batch, ages_batch, occupations_batch, 
                            movie_ids_batch, genres_batch)
        
        yhat = torch.clamp(yhat, 0.0, 1.0)

        # back to star ratings
        y = y * 5.0
        yhat = torch.ceil(yhat * 5.0)

        # add to list
        y_list.extend(y.tolist())
        yhat_list.extend(yhat.tolist())

    values = ['1', '2', '3', '4', '5']
    chart_path = os.path.join(CHART_DIR, 'movie_rating_confusion_matrix_all_data.png')
    make_matrix(y_list, yhat_list, values, 'Movie Rating Confusion Matrix - All Data', chart_path)


def make_matrix(y_true, y_pred, matrix_values, figure_title, figure_path):
    # Build confusion matrix
    print(f'Confusion Matrix: {figure_title}')
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                            index = [matrix_values],
                            columns = [matrix_values])
    plt.figure(figsize = (16,12))
    plt.subplots_adjust(bottom=0.25)
    plt.title(figure_title)
    hm = sn.heatmap(df_cm, annot=True, linewidths=.5, cmap='plasma', fmt='.2f', linecolor='grey')
    hm.set(xlabel='Predicted', ylabel='Truth')
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f'Saved: {figure_path}')


def plot_ratings():
    movie_data = MovieDataset()
    ratings = movie_data.ratings
    rate_counts = ratings['rating'].value_counts(sort=True) * 5.0
    plt.figure(figsize = (16,12))
    plt.subplots_adjust(bottom=0.25)
    plt.title('Rating Counts')
    hm = sn.barplot(x=rate_counts.index, y=rate_counts.values, palette='plasma')
    hm.set(xlabel='Rating', ylabel='Count')
    plt.savefig(os.path.join(CHART_DIR, 'movie_ratings.png'), dpi=300, bbox_inches="tight")


def make_epoch_chart(data, title, ylabel, figure_name, show=False):
    plt.figure(figsize=(6, 4))

    for d in data:
        plt.plot(data[d], label=d)

    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend()
    plt.subplots_adjust(left=0.15)
    plt.savefig(os.path.join(CHART_DIR, f'{figure_name}.png'), dpi=300, bbox_inches="tight")
    if show:
        plt.show()