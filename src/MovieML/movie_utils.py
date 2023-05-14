import matplotlib.pyplot as plt
import pandas as pd

from imdb import Cinemagoer
from io import BytesIO
from PIL import Image
import os, requests

from .movie_globals import *

def get_image_listing():
    image_list = os.listdir(IMAGE_DIR)
    image_list = [x for x in image_list if x.endswith('.jpg')]
    return image_list


def get_poster_url(code):
    id = f'{code:07d}'
    poster_path = os.path.join(POSTER_DIR, f'{id}.jpg') 
    # Check to see if I already have it
    if not os.path.isfile(poster_path):
        print('Fetching poster from IMDb...')
        # creating instance of IMDb
        ia = Cinemagoer()
        # getting information
        series = ia.get_movie(code)
        # getting cover url of the series
        cover = series.data['cover url']
        # Get image data, and save it as imdb_id
        ext = cover.split('.')[-1]
        response = requests.get(cover)
        img = Image.open(BytesIO(response.content))   
        # Get file extension
        poster_path = os.path.join(POSTER_DIR, f'{id}.{ext}') 
        img.save(poster_path)

    return poster_path


def print_predictions_no_sim(user_id, sorted_orig_ratings, sorted_new_ratings, max_rated, max_predict):
    print()
    print('*'*30)
    print(f'Top {max_rated} rated movies for user {user_id}')
    for i in range(max_rated):
        m = sorted_orig_ratings[i]
        m_rating = m[1]['rating']
        m_title = m[1]['title']
        print(f"{i+1}) Title: {m_title} :: Rating: {m_rating} ")

    print()
    print('*'*30)
    print(f'Top {max_predict} predicted movies for user {user_id}')
    predict_count = 0
    for m in sorted_new_ratings:
        m_rating = m[1]['rating']
        m_title = m[1]['title']
        # small bodge to keep overly ambitious movies from intruding
        if m_rating <= 5.0:
            print(f"{predict_count+1}) Title: {m_title} :: Predicted Rating: {m_rating:.3f} ")
            predict_count+=1
        if predict_count > max_predict:
            break
    print()
    print()


def print_predictions_with_sim(user_id, sorted_orig_ratings, predicted_similarity_sorted, predicted_top, max_rated, max_predict):
    print()
    print('*'*30)
    print(f'Top {max_rated} rated movies for user {user_id}')
    rated_images = list()
    for i in range(max_rated):
        m = sorted_orig_ratings[i]
        m_rating = m[1]['rating']
        m_title = m[1]['title']
        rated_images.append(m[1]['poster'])
        print(f"{i+1}) Title: {m_title} :: Rating: {m_rating} ")

    print()
    print('*'*30)
    print(f'Top {max_predict} predicted movies for user {user_id}')
    predicted_images = list()
    predicted_captions = list()
    for i,k in enumerate(predicted_similarity_sorted[:max_predict]):
        predicted_images.append(predicted_top[k]['poster'])
        predicted_captions.append(f'{k:.3f}')
        print(f'{i+1}) Title: {predicted_top[k]["title"]} :: Predicted Rating: {predicted_top[k]["rating"]:.1f} :: Poster Similarity: {k*100:.2f}% ')
    print()
    print()
    display_images(rated_images, predicted_images, predicted_captions)


def display_images(rated_images, predicted_images, predicted_captions):
    # Data for the images (replace with your own data)
    top_row_images = rated_images  # Replace with your actual image data
    bottom_row_images = predicted_images  # Replace with your actual image data
    bottom_row_captions = predicted_captions  # Replace with your actual captions

    # Create the subplots
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

    # Remove the unused subplots in the top row
    for ax in axes[0, 3:]:
        ax.remove()

    # Plot the images in the top row
    for i, image in enumerate(top_row_images):
        img = plt.imread(image)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')

    # Set titles for the sections
    axes[0, 0].set_title('Top User Rated Movies')
    axes[1, 0].set_title('Top Predictions/Poster Similarity')

    # Plot the images and captions in the bottom row
    for i, image in enumerate(bottom_row_images):
        img = plt.imread(image)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(bottom_row_captions[i])

    # Remove the empty subplot in the top-left corner
    axes[0, 0].axis('off')

    # Adjust the spacing between subplots
    fig.tight_layout(pad=5.0)

    plt.savefig(os.path.join(CHART_DIR, 'sim_predictions.png'), dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # print(get_image_listing())
    # print(len(get_image_listing()))
    # get_poster_url()
    pass