import torch

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from movie_dataset import MovieDataset
from movie_model import MovieRecommender
from movie_globals import *
from movie_utils import get_image_listing, get_poster_url
from movie_poster_sim import *

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

def rate_unrated_movies(user_id):
    movie_data = MovieDataset()

    user_id = 5360

    user_ratings = movie_data.get_user_rating_indicies(user_id)
    user_unrated = movie_data.get_user_unrated_movies(user_id)
    print('#'*80)
    print(f'User {user_id} has rated {len(user_ratings)} movies and has {len(user_unrated)} unrated movies')

    ###########################################################
    # get rated movies for user
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
        movie['apparent_id'] = m
        movie_ratings[m] = movie
    # sort by rating
    sorted_new_ratings = sorted(movie_ratings.items(), key=lambda x: x[1]['rating'], reverse=True)
    ###########################################################

    rated_posters = list()
    for r in sorted_orig_ratings[:3]:
        print(f'{r[1]["rating"]:.1f} :: {r[1]["title"]}')
        if r[1]['link'] != 0:
            print('fetching image...')
            rated_posters.append(get_poster_url(r[1]['link']))

    print(rated_posters)
    cluster_center = get_cluster_center(rated_posters)

    print()

    n = 0
    unrated_posters = list()
    unrated_top = dict()
    for m in sorted_new_ratings:
        rating = m[1]['rating']
        # This is an admitted bodge where some movie are getting >5.0 ratings and dominating the lists
        if rating <= 5.0:
            title = m[1]['title']
            print(f'{m[1]["rating"]:.1f} :: {m[1]["title"]}')
            sim = 0
            if m[1]['link'] != 0:
                print('fetching image...')
                p = get_poster_url(m[1]['link'])
                p_vec = get_vector(p)
                sim = get_similarity(cluster_center, p_vec)
            unrated_top[float(sim)] = {'title':title, 'rating':rating}
            
            n+=1

        if n > 20:
            break
    
    sim_keys = sorted(unrated_top.keys(), reverse=True)
    for k in sim_keys:
        print(f'{k:.3f} :: {unrated_top[k]["rating"]:.1f} :: {unrated_top[k]["title"]}')
    
    return
    for p in unrated_posters:
        p_vec = get_vector(p)
        sim = get_similarity(cluster_center, p_vec)
        print(f'{float(sim):.3f} :: {p}')
    
    return

    # get available images
    image_list = get_image_listing()

    print()
    print('*'*30)
    print(f'Top 3 rated movies for user {user_id}')
    for i in range(3):
        print(m)
        m = sorted_orig_ratings[i]
        m_id = m[1]['movie']['movie_id']
        m_rating = m[1]['rating']
        m_title = m[1]['movie']['title']
        m_genre = m[1]['movie']['genres_orig']
        m_link = m[1]['movie']['link']
        m_image = f'tt{m_link:07d}.jpg'
        print(f"{i+1}) {m_id} :: {m_rating*5.0:.1f} :: {m_title}, link: {m_image}")
        if m_image in image_list:
            print('Image found')

    print('*'*30)
    print()
    
    # print top 10
    print('*'*30)
    print(f'Top 10 movies for user {user_id}')
    for i in range(10):
        m = sorted_new_ratings[i]
        print(m)
        m_id = m[1]['movie']['movie_id']
        m_rating = m[1]['rating']
        m_title = m[1]['movie']['title']
        m_genre = m[1]['movie']['genres_orig']
        m_link = m[1]['movie']['link']
        m_image = f'tt{m_link:07d}.jpg'
        print(f"{i+1}) {m_id} :: {m_rating:.1f} :: {m_title}, link: {m_image}")
        if m_image in image_list:
            print('Image found')
    print('*'*30)

##########################################################################################
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
    rate_unrated_movies(user_id)