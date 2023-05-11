import torch

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from movie_dataset import MovieDataset
from movie_model import MovieRecommender
from movie_globals import *

def get_model(movie_data):
    # define counts for embedding layers
    num_users = movie_data.num_users
    num_ages = movie_data.num_ages
    num_genders = movie_data.num_genders
    num_occupations = movie_data.num_occupations
    num_movies = movie_data.num_movies
    num_genres = movie_data.num_genres

    # define hyperparameters
    embedding_size = 50
    hidden_size = 10

    # My GPU doesn't have enough memory to train this model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # create model and move to device
    model = MovieRecommender(num_users, num_genders, num_ages, num_occupations,
                            num_movies, num_genres,
                            embedding_size, hidden_size).to(device)

    return model

def predict_single(model, movie_data, n, device):
     # get rating data piece
    d = movie_data[n]
    #print(d)
    # X
    user_ids_batch = torch.tensor(d['user_id']).to(device)
    genders_batch = torch.tensor(d['gender']).to(device)
    ages_batch = torch.tensor(d['age']).to(device)
    occupations_batch = torch.tensor(d['occupation']).to(device)
    movie_ids_batch = torch.tensor(d['movie_id']).to(device)
    genres_batch = torch.tensor(d['genre']).to(device)
    # Y
    y =  d['rating']
    # forward pass
    yhat = model(user_ids_batch, genders_batch, ages_batch, occupations_batch, 
                    movie_ids_batch, genres_batch)
    
    return y, yhat

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

        # if n == 0:
        #     break
    
    values = ['1', '2', '3', '4', '5']
    chart_path = os.path.join(CHART_DIR, 'movie_rating_confusion_matrix_all_data.png')
    make_matrix(y_list, yhat_list, values, 'Movie Rating Confusion Matrix - All Data', chart_path)

def predict_user(user_id):
    # evaluate the model
    movie_data = MovieDataset()

    # get users
    pred_list = movie_data.get_user_rating_indicies(user_id)

    device = torch.device('cpu')
    model = get_model(movie_data)
    model.load_state_dict(torch.load(MODEL_STATE))
    model.eval()

    y_list = list()
    yhat_list = list()

    print('*'*30)
    print(f'Predicting for user {user_id}')
    for p in pred_list:
        y, yhat = predict_single(model, movie_data, p, device)
        
        # back to star ratings
        y = int(float(y) * 5.0)
        yhat = round(float(yhat) * 5.0)

        y_list.append(y)
        yhat_list.append(yhat)

        print(f'Item: {p} -> Predicted: {round(yhat)} Actual: {int(y)}')
    print('*'*30)


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

if __name__ == '__main__':
    matrix_all()