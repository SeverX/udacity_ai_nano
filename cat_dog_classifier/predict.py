#Predict flower name from an image with predict.py along with the probability of that name. 
#That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

#Basic usage: python predict.py /path/to/image checkpoint
#Options:
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu

#python predict.py --image_path flowers/test/2/image_05109.jpg
#python predict.py --image_path flowers/test/2/image_05109.jpg --checkpoint_path save/checkpoint.pth --gpu --top_k 7  

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import time
from PIL import Image
import argparse

DEFAULT_TEST_IMAGE = 'flowers/test/2/image_05109.jpg'
DEFAULT_TOP_K = 5
DEFAULT_GPU = True
SIZE_CROP = 224
SIZE_RESIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def parse_input_arguments():
    '''
    Parse input arguments
    Arguments:
        None
    Returns:
        image_path (str): path to image for classification
        checkpoint_path (str): checkpoint path to load the model
        top_k (int): number of top clases to use
        category_names (str): Mapping file with categories and labels
        gpu (boolean): Enable GPU
    '''
    parser = argparse.ArgumentParser(description = "Predict using a deep neural network")
    parser.add_argument('--image_path', type = str, default = DEFAULT_TEST_IMAGE, help = 'Dataset path')
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoint.pth', help = 'Path to load trained model checkpoint')
    parser.add_argument('--top_k', type = int, default = DEFAULT_TOP_K, help = 'Top K most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'File .json for the mapping of categories to real names')
    parser.add_argument('--gpu', action = "store_true", default = DEFAULT_GPU, help = 'Use GPU if available')

    args = parser.parse_args()
    #print(args)

    return args.image_path, args.checkpoint_path, args.top_k, args.category_names, args.gpu


def load_model_checkpoint(file_path):
    '''
    Load the model checkpoint
    Arguments:
        file_path (str): checkpoint file path
    Returns:
        model (object): model loaded from checkpoint
    '''
    checkpoint = torch.load(file_path)
    model = getattr(torchvision.models, checkpoint['network'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(pil_image):
    '''
    Transform a PIL image for a PyTorch model
    Arguments:
        pil_image (PIL.Image): PIL image 
    Returns:
        np_image (numpy.array): image in numpy array
    '''
    img_loader = transforms.Compose([transforms.Resize(SIZE_RESIZE),
                                    transforms.CenterCrop(SIZE_CROP), 
                                    transforms.ToTensor()])
    
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array(NORMALIZE_MEAN)
    std = np.array(NORMALIZE_STD)
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean) / std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image


def get_prediction(image_path, model, top_k_probabilities = DEFAULT_TOP_K):
    '''
    Predict highly likely categories for an image using a trained deep learning model
    Arguments:
        image_path (str): path to the image
        model (object): model to make predictions with
        top_k_probabilities (int): number of top clases to show
    
    Returns:
        top_p (list): top k probabilities
        top_mapped_classes (list): top k label classes
    '''
    # Use GPU if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model.to(device)
    model.eval()

    pil_image = Image.open(image_path)
    
    np_image = process_image(pil_image)
    tensor_image = torch.from_numpy(np_image)
    
    inputs = Variable(tensor_image)
    
    if torch.cuda.is_available():
        inputs = Variable(tensor_image.float().cuda())           
        
    inputs = inputs.unsqueeze(dim = 0)
    log_ps = model.forward(inputs)
    ps = torch.exp(log_ps)    

    top_p, top_classes = ps.topk(top_k_probabilities, dim = 1)
    
    class_to_idx_inverted = {model.class_to_idx[c]: c for c in model.class_to_idx}
    top_mapped_classes = list()
    
    for label in top_classes.cpu().detach().numpy()[0]:
        top_mapped_classes.append(class_to_idx_inverted[label])
    
    top_p = top_p.cpu().detach().numpy()[0]
    
    return top_p, top_mapped_classes


def get_cat_to_name(category_names):
    '''
    Load json mapping file for categories with labels
    Arguments:
        category_names (str): Mapping file category label to name
    Returns:
       category_label_to_name (dict): dictionary for mapping category label to name
    '''
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name


def load_model(category_names, checkpoint_path):
    '''
    Load model from checkpoint file
    Arguments:
        category_names (str): mapping filee
        checkpoint_path (str): checkpoint path
    Returns:
        model (object): model to make predictions
        cat_to_name (dict): dictionary for mapping category label to name
    '''

    print('Load the model checkpoint from {}'.format(checkpoint_path))
    model = load_model_checkpoint(checkpoint_path)

    print('Load category name and label mapping')
    cat_to_name = get_cat_to_name(category_names)

    return model, cat_to_name


def predict(image_path, checkpoint_path, top_k, category_names, gpu):
    '''
    Predict category(ies) for an image
    Arguments:
        image_path (str): image path to classify
        checkpoint_path (str): checkpoint path
        top_k (int): number of top clases to use
        category_names (str): mapping file category label to name
        gpu (boolean): Enable GPU
    Returns:
       None
    '''
    model, cat_to_name = load_model(category_names, checkpoint_path)

    top_p, top_classes = get_prediction(image_path, model, top_k_probabilities = top_k)
    print('Probabilities: ', top_p)
    print('Categories:    ', [cat_to_name[c] for c in top_classes])

    #if image_path == DEFAULT_TEST_IMAGE:
        # we know the directory structure so the penultimate component of the path is the categoryh path/category/image.jpg
    path_parts = image_path.split('/')
    real_category = path_parts[-2] 
    print('True category: ', cat_to_name[real_category])


if __name__ == "__main__":
    image_path, checkpoint_path, top_k, category_names, gpu = parse_input_arguments()   
    predict(image_path, checkpoint_path, top_k, category_names, gpu)