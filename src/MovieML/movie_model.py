import torch
import torch.nn as nn


# Define the model architecture
class MovieRecommender(nn.Module):
    def __init__(self, num_users, num_genders, num_ages, num_occupations, num_movies, num_genres, embedding_dim, hidden_dim):
        super(MovieRecommender, self).__init__()
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