import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

import pandas as pd
import matplotlib.pyplot as plt

from movie_dataset import MovieDataset
from movie_model import MovieRecommender
from movie_globals import *
from movie_utils import make_epoch_chart



def main(epoch_count=5, batch_size=1000, validation_ratio=0.2, 
         embedding_count=50, hidden_count=10, eta=0.001):
    # Train the model
    movie_data = MovieDataset()

    vaid_ratio = validation_ratio

    # create training and validation datasets
    num_movies = len(movie_data)
    num_validation = int(vaid_ratio * num_movies)
    num_train = num_movies - num_validation

    # split those out...
    train_data, valid_data = random_split(movie_data, [num_train, num_validation])

    # create data loaders
    num_movies = batch_size
    movie_train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=num_movies,
                                                shuffle=True,
                                                num_workers=4)

    movie_valid_loader = torch.utils.data.DataLoader(valid_data,
                                                batch_size=num_movies,
                                                shuffle=True,
                                                num_workers=4)

    # define counts for embedding layers
    num_users = movie_data.num_users
    num_ages = movie_data.num_ages
    num_genders = movie_data.num_genders
    num_occupations = movie_data.num_occupations
    num_movies = movie_data.num_movies
    num_genres = movie_data.num_genres

    # define hyperparameters
    num_epochs = epoch_count
    embedding_size = embedding_count
    hidden_size = hidden_count

    # My GPU doesn't have enough memory to train this model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # create model and move to device
    model = MovieRecommender(num_users, num_genders, num_ages, num_occupations,
                            num_movies, num_genres,
                            embedding_size, hidden_size).to(device)
    
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=eta)

    report = str(model)
    print(report)

    print('Training model...')
    continue_training = True

    epoch_train_losses = []
    epoch_valid_losses = []

    for epoch in range(num_epochs):
        running_train_loss = 0.0
        running_valid_loss = 0.0
        
        # TRAINING
        model.train()
        print(f'Training epoch {epoch+1}...')
        for n,d in enumerate(movie_train_loader):
            print('.', end='', flush=True)
            # X
            user_ids_batch = d['user_id'].to(device)
            genders_batch = d['gender'].to(device)
            ages_batch = d['age'].to(device)
            occupations_batch = d['occupation'].to(device)
            movie_ids_batch = d['movie_id'].to(device)
            genres_batch = d['genre'].to(device)
            # Y
            ratings_batch =  d['rating'].to(device)
            
            optimizer.zero_grad()
            outputs = model(user_ids_batch, genders_batch, ages_batch, occupations_batch, 
                             movie_ids_batch, genres_batch)
            
            # RMSE loss
            loss = torch.sqrt(criterion(outputs, ratings_batch))

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        print()
        epoch_train_losses.append(running_train_loss / float(num_movies))
        epoch_train_loss = epoch_train_losses[-1]

        # VALIDATION
        model.eval()
        with torch.no_grad():
            print(f'\nValidating epoch {epoch+1}...')
            for n,d in enumerate(movie_valid_loader):
                print('.', end='', flush=True)
                # X
                user_ids_batch = d['user_id'].to(device)
                genders_batch = d['gender'].to(device)
                ages_batch = d['age'].to(device)
                occupations_batch = d['occupation'].to(device)
                movie_ids_batch = d['movie_id'].to(device)
                genres_batch = d['genre'].to(device)
                # Y
                ratings_batch =  d['rating'].to(device)
                # forward pass
                outputs = model(user_ids_batch, genders_batch, ages_batch, occupations_batch, 
                                movie_ids_batch, genres_batch)
                
                # RMSE loss
                loss = torch.sqrt(criterion(outputs, ratings_batch))
                running_valid_loss += loss.item()

        print()
        epoch_valid_losses.append(running_valid_loss / float(num_movies))
        epoch_valid_loss = epoch_valid_losses[-1]

        # print epoch results
        epoch_results = '*'*30 + '\n'
        epoch_results += f'Epoch {epoch+1}/{num_epochs}\n'
        epoch_results += f'Training loss: {epoch_train_loss:.4f}\n'
        epoch_results += f'Validation loss: {epoch_valid_loss:.4f}\n'
        if epoch > 0:
            epoch_results += f'Training loss difference: {epoch_train_losses[-1] - epoch_train_losses[-2]:.4f}\n'
            epoch_results += f'Validation loss difference: {epoch_valid_losses[-1] - epoch_valid_losses[-2]:.4f}\n'
        epoch_results += '*'*30 + '\n\n'
        print(epoch_results)
        report += epoch_results

        # store losses to dataframe
        df = pd.DataFrame({'train_loss': epoch_train_losses,
                           'valid_loss': epoch_valid_losses})
        df.to_csv(os.path.join(DATA_OUT_DIR, 'movie_losses.csv'), index=False)

        make_epoch_chart(data=df, title='RSME Loss', ylabel='RSME', figure_name='movie_recco_loss.png', show=False)

        save_report = ''
        # if validation loss decreases, save model
        if epoch > 0 and epoch_valid_losses[-2] > epoch_valid_losses[-1]:
            model_path = os.path.join(MODEL_DIR, 'movie_model.pth')
            torch.save(model.state_dict(), model_path)
            save_report += f'Saved model to {model_path}'
            print(save_report)
        report += save_report

        stop_report = ''
        # if validation loss increases 4 times in a row, stop training
        if epoch > 5 and epoch_valid_losses[-4] < epoch_valid_losses[-3] < epoch_valid_losses[-2] < epoch_valid_losses[-1]:
            stop_report = 'Stopping training...'
            stop_report += f'Best model at epoch {epoch-3}'
            stop_report += f'Best validation loss: {epoch_valid_losses[epoch-3]}'
            stop_report += f'Best training loss: {epoch_train_losses[epoch-3]}'
            stop_report += f'Last validation loss: {epoch_valid_losses[epoch]}'
            stop_report += f'Last training loss: {epoch_train_losses[epoch]}'
            report += stop_report
            print(stop_report)
        report += stop_report

        
        # write out report and stop training if there is a stop report
        if stop_report != '':
            # write report to file in DATA_OUT_DIR
            with open(os.path.join(DATA_OUT_DIR, 'movie_report.txt'), 'w') as f:
                f.write(report)
            # stop training
            break
        

    # Make recommendations
    # user_id = 1  # example user ID
    # unwatched_movies = [i for i in range(num_movies) if user_item_matrix[user_id, i] == 0]
    # user_ids = torch.tensor([user_id] * len(unwatched_movies)).to(device)
    # item_ids = torch.tensor(unwatched_movies).to(device)
    # with torch.no_grad():
    #     predicted_ratings = model(user_ids, item_ids)
    # recommended_movies = np.argsort(-predicted_ratings)[:20]

if __name__ == '__main__':
    main()