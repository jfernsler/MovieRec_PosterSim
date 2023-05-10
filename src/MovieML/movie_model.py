import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from movie_dataset import MovieDataset

# Define the model architecture
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_genders, num_ages, num_occupations, num_movies, num_genres, embedding_dim, hidden_dim):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)

        self.fc1 = nn.Linear(6 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

        self.double()

    def forward(self, user_ids, gender_ids, age_ids, occupation_ids, movie_ids, genre_ids):
        user_embedded = self.user_embedding(user_ids)
        gender_embedded = self.gender_embedding(gender_ids)
        age_embedded = self.age_embedding(age_ids)
        occupation_embedded = self.occupation_embedding(occupation_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        genre_embedded = self.genre_embedding(genre_ids)

        x = torch.cat([user_embedded,
                              gender_embedded,
                              age_embedded,
                              occupation_embedded,
                              movie_embedded,
                              genre_embedded], dim=-1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1)

def main():
    # Train the model
    movie_data = MovieDataset()
    num_movies = 5
    movie_loader = torch.utils.data.DataLoader(movie_data,
                                                batch_size=num_movies,
                                                shuffle=True,
                                                num_workers=4)

    embedding_size = 50
    hidden_size = 10
    num_users = movie_data.num_users
    num_ages = movie_data.num_ages
    num_genders = movie_data.num_genders
    num_occupations = movie_data.num_occupations
    num_movies = movie_data.num_movies
    num_genres = movie_data.num_genres

    num_epochs = 5

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')


    model = RecommenderNet(num_users, num_genders, num_ages, num_occupations,
                            num_movies, num_genres,
                            embedding_size, hidden_size).to(device)
    
    criterion = nn.MSELoss().to(device)
    #criterion = nn.RMSELoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model)

    print('Training model...')

                # 'user_id': user_id,
                # 'movie_id': movie_id,
                # 'gender': gender,
                # 'age': age,
                # 'occupation': occupation,
                # 'movie': movie, 
                # 'genre': genre,
                # 'link': movie_link, 
                # 'rating': rating

    for epoch in range(num_epochs):
        running_loss = 0.0
        for n,d in enumerate(movie_loader):
            print(f'BATCH {n}')
            # X
            user_ids_batch = d['user_id'].to(device)
            genders_batch = d['gender'].to(device)
            ages_batch = d['age'].to(device)
            occupations_batch = d['occupation'].to(device)
            movie_ids_batch = d['movie_id'].to(device)
            genres_batch = d['genre'].to(device)
            # Y
            ratings_batch =  d['rating'].to(device)

            # print(f'User IDs : {user_ids_batch}')
            # print(f'Genders : {genders_batch}')
            # print(f'Ages : {ages_batch}')
            # print(f'Occupations : {occupations_batch}')
            # print(f'Movie IDs : {movie_ids_batch}')
            # print(f'Genres : {genres_batch}')
            # print(f'Ratings : {ratings_batch}')
            
            optimizer.zero_grad()
            outputs = model(user_ids_batch, genders_batch, ages_batch, occupations_batch, 
                                movie_ids_batch, genres_batch)
            # try:
            #     outputs = model(user_ids_batch, genders_batch, ages_batch, occupations_batch, 
            #                     movie_ids_batch, genres_batch)
            # except:
            #     print(f'User IDs : {user_ids_batch}')
            #     print(f'Gender : {genders_batch}')
            #     print(f'Age : {ages_batch}')
            #     print(f'Occupation : {occupations_batch}')
            #     print(f'Movie IDs : {movie_ids_batch}')
            #     print(f'Genres : {genres_batch}')
            #print(f'Outputs : {outputs}')
            loss = criterion(outputs, ratings_batch).to(device)
            #print(f'Loss : {loss}')
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(num_movies)

        print('Epoch {}/{} Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

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