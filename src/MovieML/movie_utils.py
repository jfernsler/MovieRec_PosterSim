import matplotlib.pyplot as plt
import pandas as pd

from imdb import Cinemagoer
from io import BytesIO
from PIL import Image
import os, requests

from movie_globals import *

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


def get_image_listing():
    image_list = os.listdir(IMAGE_DIR)
    image_list = [x for x in image_list if x.endswith('.jpg')]
    return image_list

def save_poster(imdb_id, img_url):
    '''
    Function that fetches and save the poster image from provided url
    and saves it with the provided id (corresponding with IMDb).
    Won't replace (or even fetch) if file already exists.
    
    INPUT:  id from imdb, url where to find image
    OUTPUT: boolean flag if saved or not.
    '''
    import os.path
    
    # Get file extension
    ext = img_url.split('.')[-1]
    
    # Check to see if I already have it
    if os.path.isfile(f'posters/{imdb_id}.{ext}'):
        return False
    
    # Get image data, and save it as imdb_id
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))   

    poster_path = os.path.join(POSTER_DIR, f'{imdb_id}.{ext}') 
    img.save(poster_path)
    
    return poster_path

def get_poster_url(code):

    id = f'{code:07d}'
    poster_path = os.path.join(POSTER_DIR, f'{id}.jpg') 

    # Check to see if I already have it
    if not os.path.isfile(poster_path):
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
    



if __name__ == '__main__':
    # print(get_image_listing())
    # print(len(get_image_listing()))
    get_poster_url()