import torch

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from .movie_dataset import MovieDataset
from .movie_model import MovieRecommender
from .movie_globals import *
from .movie_utils import get_image_listing, get_poster_url, print_predictions_no_sim, print_predictions_with_sim
from .movie_poster_sim import *

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
    """
    Predict a users ratings for a previously rated movie movie, and 
    return the actual rating and the predicted rating
    """
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

def predict_user_movie(model, user, movie, device):
    """
    Predict a user's rating for a movie and return the predicted rating
    """
    # user X
    user_ids_batch = torch.tensor(user['user_id']).to(device)
    genders_batch = torch.tensor(user['gender']).to(device)
    ages_batch = torch.tensor(user['age']).to(device)
    occupations_batch = torch.tensor(user['occupation']).to(device)
    # movie X
    movie_ids_batch = torch.tensor(movie['movie_id']).to(device)
    genres_batch = torch.tensor(movie['genre']).to(device)
    # forward pass
    yhat = model(user_ids_batch, genders_batch, ages_batch, occupations_batch, 
                    movie_ids_batch, genres_batch)
    
    return yhat

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


def predict_movies_to_user(user_id, similarity=True, max_predict=30, max_sim=5, max_rated=3):
    movie_data = MovieDataset()

    #user_id = 5360

    user_ratings = movie_data.get_user_rating_indicies(user_id)
    user_unrated = movie_data.get_user_unrated_movies(user_id)

    print()
    print('#'*20, 'Begin Predictions', '#'*20)
    print(f'User {user_id} has rated {len(user_ratings)} movies and has {len(user_unrated)} unrated movies')

    ###########################################################
    # get rated movies for user
    # no predictions, just data
    rated = movie_data.get_user_rating_indicies(user_id)
    rated_movies = dict()
    for r in rated:
        rating = movie_data.ratings.loc[r]
        movie = movie_data.get_movie(rating['movieID'])
        movie['rating' ] = rating['rating'] * 5.0
        rated_movies[r] = movie
    # sort rated movies by rating
    sorted_orig_ratings = sorted(rated_movies.items(), key=lambda x: x[1]['rating'], reverse=True)
    ###########################################################

    print(f'Predicting ratings for {len(user_unrated)} movies')

    ###########################################################
    # predict ratings from trained model
    device = torch.device('cpu')
    model = get_model(movie_data)
    model.load_state_dict(torch.load(MODEL_STATE))
    model.eval()

    user = movie_data.get_user(user_id)
    
    movie_ratings = dict()
    # predict ratings for unrated movies
    for m in user_unrated:
        movie = movie_data.get_movie(m)
        rating = predict_user_movie(model, user, movie, device)
        movie['rating'] = float(rating * 5.0)
        movie_ratings[m] = movie
    # sort by predicted ratings
    sorted_new_ratings = sorted(movie_ratings.items(), key=lambda x: x[1]['rating'], reverse=True)
    ###########################################################

    # if no similarity, just print the predictions
    # printing 3x the max_sim to account for over weighted movies
    if not similarity:
        print_predictions_no_sim(user_id, sorted_orig_ratings, sorted_new_ratings, max_rated, max_sim*3)
        return

    print('Finding posters - be patient please...')

    ###########################################################
    # fetch posters for rated movies
    rated_posters = list()    # keep a list just for the cluster center
    for r in sorted_orig_ratings[:max_rated]:
        #print(f'{r[1]["rating"]:.1f} :: {r[1]["title"]}')
        if r[1]['link'] != 0:
            r[1]['poster'] = get_poster_url(r[1]['link'])
            rated_posters.append(r[1]['poster'])
        else:
            r[1]['poster'] = NO_POSTER
            rated_posters.append(NO_POSTER)
    # generate cluster center movie poster embeddings
    cluster_center = get_cluster_center(rated_posters)
    ###########################################################


    ###########################################################
    # fetch posters for predicted movies
    predict_count = 0
    predicted_top = dict()
    for m in sorted_new_ratings:
        rating = m[1]['rating']
        # This is an admitted bodge where some movie are getting >5.0 ratings and dominating the lists
        if rating <= 5.0:
            title = m[1]['title']
            #print(f'{m[1]["rating"]:.1f} :: {m[1]["title"]}')
            sim = 0
            if m[1]['link'] != 0:
                p = get_poster_url(m[1]['link'])
                p_vec = get_vector(p)
                sim = get_similarity(cluster_center, p_vec)
            else:
                p = NO_POSTER
                sim = 0
            # store dict with similarity as key
            predicted_top[float(sim)] = {'title':title, 'rating':rating, 'poster':p}
            predict_count+=1

        if predict_count > max_predict:
            break

    predicted_similarity_sorted = sorted(predicted_top.keys(), reverse=True)
    ###########################################################

    print_predictions_with_sim(user_id, sorted_orig_ratings, predicted_similarity_sorted, predicted_top, max_rated, max_sim)

    return
    for k in predicted_similarity_sorted[:max_sim]:
        print(f'{k:.3f} :: {predicted_top[k]["rating"]:.1f} :: {predicted_top[k]["title"]}')
    
    return
    
"""
TODO: 
    1. for any given user, display the 3 most recently watched, highest rated movies
    2. for any given user, display the 10 highest predicted movies
    3. find the cluster center for the 3 most recently watched, highest rated movies
    4. find the 3 most similar movies to the cluster center
    5. display all of that nicely.
"""




if __name__ == '__main__':
    #matrix_all()
    #plot_ratings()
    # random number between 1 and 6040
    user_id = random.randint(1, 6040)
    predict_movies_to_user(user_id, similarity=True)