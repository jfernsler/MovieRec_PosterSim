import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

import matplotlib.pyplot as plt


from movie_globals import *
import os, random

def get_vec_model_and_transforms():
    # Load the pretrained model
    model = models.resnet18(weights='DEFAULT')
    # define layer for feature extraction
    layer = model._modules.get('avgpool')
    # Set model to evaluation mode
    model.eval()
    # image transforms for resnet
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    return {'model': model, 'layer': layer, 'scaler': scaler, 'normalize': normalize, 'to_tensor': to_tensor}

def get_vector(image_name):
    model_dict = get_vec_model_and_transforms()
    model = model_dict['model']
    layer = model_dict['layer']
    scaler = model_dict['scaler']
    normalize = model_dict['normalize']
    to_tensor = model_dict['to_tensor']
    # get image into correct format
    img = Image.open(image_name)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # create an embedding tensor
    my_embedding = torch.zeros(512)
    # function to copy data out of the layer will be a callback
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # attach it to the layer
    h = layer.register_forward_hook(copy_data)
    # evaluate the image
    model(t_img)
    # remove the copy function from the layer
    h.remove()

    return my_embedding

def get_images(count=10):
    image_list = os.listdir(IMAGE_DIR)
    image_list = [x for x in image_list if x.endswith('.jpg')]
    image_list = random.sample(image_list, count)
    return image_list


def display_images(image_list):
    fig = plt.figure(figsize=(10, 10))
    for n, image_name in enumerate(image_list):
        ax = fig.add_subplot(1, len(image_list), n+1)
        ax.imshow(Image.open(os.path.join(IMAGE_DIR, image_name)))
    plt.show()

def display_images_with_sim(primary_image, image_list, sim_list):  
    # place images in a grid with and additional image centered up top
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(Image.open(os.path.join(IMAGE_DIR, primary_image)))
    ax.axis('off')
    ax.set_title('Primary Image')
    for n, (image_name, sim) in enumerate(zip(image_list, sim_list)):
        ax = fig.add_subplot(2, len(image_list), n+1+len(image_list))
        ax.imshow(Image.open(os.path.join(IMAGE_DIR, image_name)))
        ax.axis('off')
        ax.set_title('{:.3f}'.format(sim))
    plt.show()


def get_similarity(primary_vec, compare_vec):
    # Calculate the similarity between 2 vectors
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    sim = cosine_sim(primary_vec.unsqueeze(0),
                    compare_vec.unsqueeze(0))
    return sim

def get_cluster_center(img_list):
    # get the cluster center for a set of images
    model_dict = get_vec_model_and_transforms()
    img_vecs = [get_vector(image_name) for image_name in img_list]
    img_vecs = torch.stack(img_vecs)
    return torch.mean(img_vecs, dim=0)


def main():
    # get a random set of images and compare the first image to the rest
    model_dict = get_vec_model_and_transforms()
    image_list = get_images()

    image_vecs = [get_vector(os.path.join(IMAGE_DIR, image_name), model_dict) for image_name in image_list]

    sim_list = list()
    for image_vec in image_vecs[1:]:
        sim = get_similarity(image_vecs[0], image_vec)
        sim_list.append(float(sim))
    
    primary_image = image_list[0]
    sim_images = image_list[1:]

    # sort sim_images and sim_list by sim_list
    sim_images, sim_list = zip(*sorted(zip(sim_images, sim_list), key=lambda x: x[1], reverse=True))

    for n,s in zip(sim_images, sim_list):
        print(n, s)

    display_images_with_sim(primary_image, sim_images[:3], sim_list[:3])

if __name__ == '__main__':
    main()
    