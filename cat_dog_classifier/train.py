# Image classifier application
# python image_classifier_project.py   

#Basic usage: python train.py data_directory
#Prints out training loss, validation loss, and validation accuracy as the network trains
#Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu
# Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

#python train.py flowers --save_dir save --arch 'vgg13' --learning_rate 0.002 --hidden_units 512 --epochs 3 --gpu 

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import json
import time
import argparse

from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torch import optim
from torch.autograd import Variable

DEFAULT_CHECKPOINT_FILENAME = 'checkpoint.pth'
DEGREES_ROTATION = 30
SIZE_CROP = 224
SIZE_RESIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 64
DROPOUT_PROBABILITY = 0.5
IN_FEATURES = 25088


def parse_input_arguments():
    '''
    Parse input arguments
    Arguments:
        None
    Returns:
        data_dir (str)  : directory with the data
        save_dir (str)  : directory for saving a checkpoint
        arch (str)      : network architecture from torchvision
        learning_rate (float)   : learning rate for optimizer
        hidden_units (int)      : hidden units to use
        epochs (int)            : epochs to train
        gpu (boolean)           : Enable the use of CUDA(GPU)
    '''
    parser = argparse.ArgumentParser(description = "Train a deep neural network")
    parser.add_argument('data_dir', type = str, default = 'flowers', help = 'Dataset path')
    parser.add_argument('--save_dir', type = str, default = None, help = 'Path to save trained model checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg16', choices = ['vgg11', 'vgg13', 'vgg16', 'vgg19'], help = 'Model architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate for optimizator')
    parser.add_argument('--hidden_units', type = int, default = 1024, help = 'Number of hidden units')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of epochs for training')
    parser.add_argument('--gpu', action = "store_true", default = True, help = 'Use GPU if available')

    args = parser.parse_args()
    return args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu


def get_data_directories(data_directory):
    '''
    Get the directories for train, validation and test data
    Arguments:
        data_directory (str): base data directory
    Returns:
        train_directory (str): directory for training dataset
        valid_directory (str): directory for validation dataset
        test_directory (str): directory for test dataset
    ''' 
    train_directory = data_directory + '/train'
    valid_directory = data_directory + '/valid'
    test_directory = data_directory + '/test'

    return train_directory, valid_directory, test_directory

def get_cat_to_name(train_directory, valid_directory, test_directory):
    '''
    Load json mapping file from category label to category name
    Returns:
       category_label_to_name (dict): dictionary - category to label
    '''
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    flower_types = len(cat_to_name)

    # Check if all folders have the same number of flower classes and as it's defined in JSON 
    if len(os.listdir(train_directory)) == len(os.listdir(valid_directory)) == len(os.listdir(test_directory)) == flower_types:
        print(f"Fowers: {flower_types}")
    else:
        print(f"Error: Different number of flower types")
        exit()

    return cat_to_name, flower_types

def load_datasets(train_directory, valid_directory, test_directory):
    '''
    Load datasets
    Arguments:
        train_directory (str): directory for training dataset
        valid_directory (str): directory for validation dataset
        test_directory (str): directory for test dataset
    Returns:
       train_data (torchvision.datasets.ImageFolder): train dataset
       valid_data (torchvision.datasets.ImageFolder): validation dataset
       test_data (torchvision.datasets.ImageFolder): test dataset
    '''
    train_transforms = transforms.Compose([transforms.RandomRotation(DEGREES_ROTATION),
                                        transforms.RandomResizedCrop(SIZE_CROP),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

    valid_transforms = transforms.Compose([transforms.Resize(SIZE_RESIZE), 
                                        transforms.CenterCrop(SIZE_CROP),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

    test_transforms = transforms.Compose([transforms.Resize(SIZE_RESIZE), 
                                        transforms.CenterCrop(SIZE_CROP),
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
                                        ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_directory, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_directory, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_directory, transform = test_transforms)

    return train_data, valid_data, test_data


def get_data_loaders(train_data, valid_data, test_data):
    '''
    Get data loaders
    Arguments:
       train_data (torchvision.datasets.ImageFolder): train dataset
       valid_data (torchvision.datasets.ImageFolder): validation dataset
       test_data (torchvision.datasets.ImageFolder): test dataset
    Returns:
       trainloader (torch.utils.data.DataLoader): train data loader
       validloader (torch.utils.data.DataLoader): validation data loader
       testloader (torch.utils.data.DataLoader): test data loader
    '''

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

    return trainloader, validloader, testloader


def get_pretrained_model(network, number_classes, hidden_units):
    '''
    Get pretrained model and adapt it to current needs
    Arguments:
        network (str): network architecture to use
        number_classes (int): number of classes
    Returns:
       model (object): model adapted to current needs
       classifier (object): classifier adapted to current needs
    '''
    model = getattr(torchvision.models, network)(pretrained = True)
    out_features = hidden_units

    #print(model)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(IN_FEATURES, out_features)),
                                            ('drop', nn.Dropout(p = DROPOUT_PROBABILITY)),
                                            ('relu', nn.ReLU()),
                                            ('fc2', nn.Linear(out_features, number_classes)),
                                            ('output', nn.LogSoftmax(dim = 1))
                                            ]))
        
    model.classifier = classifier
    #print(model)
    return model, classifier

def save_model_checkpoint(model, train_data, network, number_classes, learning_rate, classifier, epochs, optimizer, save_path, checkpoint_filename):
    '''
    Save model checkpoint
    Arguments:
        network (str): network architecture used
        number_classes (int): number of classes
    Returns:
       None
    '''
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'network': network,
                'input_size': IN_FEATURES,
                'output_size': number_classes,
                'learning_rate': learning_rate,       
                'batch_size': BATCH_SIZE,
                'classifier' : classifier,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_path)

def train(data_directory, save_directory, network, learning_rate, hidden_units, epochs, gpu):
    '''
    Train a network and save it in a checkpoint file
    Arguments:
        data_directory (str): directory were data is stored
        save_directory (str): directory to save checkpoint
        network (str): network architecture to use
        learning_rate (float): learning rate to use
        hidden_units (int): hidden units to use
        epochs (int): epochs to use
        gpu (boolean): Enable the use of GPU
    Returns:
       None
    '''
    print('Get data directories')
    train_directory, valid_directory, test_directory = get_data_directories(data_directory)

    print('Load mapping of category names to labels. Get the number of categories')
    category_label_to_name, number_classes = get_cat_to_name(train_directory, valid_directory, test_directory)

    print('Load datasets')
    train_data, valid_data, test_data = load_datasets(train_directory, valid_directory, test_directory)

    print('Get data loaders')
    trainloader, validloader, testloader = get_data_loaders(train_data, valid_data, test_data)
    
    print('Load pretrained model')
    model, classifier = get_pretrained_model(network, number_classes, hidden_units)

    # Train the network
    device = torch.device('cuda' if torch.cuda.is_available() and gpu == True else 'cpu')
    print('Device:', device)

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    #Loss function
    criterion = nn.NLLLoss()

    model.to(device)

    print('Start training')
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:     
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
           
        else:           
            valid_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    log_ps = model.forward(inputs)
                    loss = criterion(log_ps, labels)
            
                    valid_loss += loss.item()
            
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim = 1)                    
                    equals = top_class == labels.view(*top_class.shape)                    
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            model.train()
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Test accuracy: {accuracy/len(validloader):.3f}")
                    
    end_time = time.time()

    training_time = end_time - start_time
    print(f'Training time: {training_time / 60:.0f}m {training_time % 60:.0f}s')

    # Save model checkpoint
    save_path = ''

    if save_directory == None:
        save_path = DEFAULT_CHECKPOINT_FILENAME
    else:
        save_path = save_directory + '/' + DEFAULT_CHECKPOINT_FILENAME
    
    save_model_checkpoint(model, train_data, network, number_classes, learning_rate, 
                          classifier, epochs, optimizer, save_path, DEFAULT_CHECKPOINT_FILENAME)
    print(f'Save the checkpoint in {save_path}')

if __name__ == "__main__":
    data_directory, save_directory, network, learning_rate, hidden_units, epochs, gpu = parse_input_arguments()   
    train(data_directory, save_directory, network, learning_rate, hidden_units, epochs, gpu)