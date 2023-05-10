from MovieML import MovieDataset

import numpy as np
import argparse, sys

def test_dataset():
    dataset = MovieDataset()
    rand_idx_array = np.random.randint(0, len(dataset), size=10)
    for idx in rand_idx_array:
        print(dataset[idx])


def main(args):
    if args.test_dataset:
        print('*'*10, ' Test Dataset ', '*'*10)
        test_dataset()
    elif args.eval_batch:
        print('*'*10, ' Batch Image Evaluation ', '*'*10)
        pass
    elif args.eval_timing:
        print('*'*10, ' 50 Image Timing Check ', '*'*10)
        pass
    elif args.train:
        print('*'*10, ' Training On Reduced Set ', '*'*10)
        pass

if __name__=='__main__':
    test_dataset()
    sys.exit(0)

    parser = argparse.ArgumentParser(description='CS614 Assignment 2 - Movie Recommender')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--test_dataset', 
                        action='store_true', help='Check the dataset return values')
    group.add_argument('-es', '--eval_single', 
                        action='store_true', help='Evaluate a single image on the weld_resnet50_model model fine tuned for this class')
    group.add_argument('-eb', '--eval_batch', 
                        action='store_true', help='Evaluate a batch of images on the weld_resnet50_model model fine tuned for this class')
    group.add_argument('-t', '--train', 
                        action='store_true', help='Will train the model on the reduced dataset for 20 epochs as v6')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    main(args)
