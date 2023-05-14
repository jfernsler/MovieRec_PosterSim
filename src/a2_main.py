from MovieML import MovieDataset
from MovieML import predict_movies_to_user

import numpy as np
import argparse, sys, random

def test_dataset():
    dataset = MovieDataset()
    rand_idx_array = np.random.randint(0, len(dataset), size=10)
    for idx in rand_idx_array:
        print(dataset[idx])

def predict_no_similarity():
    user_id = random.randint(1, 6040)
    predict_movies_to_user(user_id, similarity=False)  

def predict_with_similarity():
    user_id = random.randint(1, 6040)
    predict_movies_to_user(user_id, similarity=True)   

def main(args):
    if args.test_dataset:
        print('*'*10, ' Test Dataset ', '*'*10)
        test_dataset()
    elif args.predict:
        print('*'*10, ' Predict From Model ', '*'*10)
        predict_no_similarity()
    elif args.predict_similar:
        print('*'*10, ' Predict From Model - Evaluate Poster Similarity ', '*'*10)
        predict_with_similarity()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CS614 Assignment 2 - Movie Recommender')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--test_dataset', 
                        action='store_true', help='Check the dataset return values')
    group.add_argument('-p', '--predict', 
                        action='store_true', help='Make a prediction for a random user based only on the trained model')
    group.add_argument('-psim', '--predict_similar', 
                        action='store_true', help='Make a prediction for a random user based on the trained model and poster similarity')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    main(args)
