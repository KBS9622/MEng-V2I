import os
import sys
import numpy as np

from torchvision.transforms import transforms
from torch.utils.data.dataset import random_split
from torchvision.datasets import CIFAR10, ImageFolder

def transformations(dataset, subset=None):
    
    if dataset == 'CIFAR10':
        ''' Image processing for CIFAR-10 '''
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        transformations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), padding=4), 
                transforms.ToTensor(), 
                normalize])
                
     # Append more custom datasets here
     
def load_dataset(data_path, dataset: str, comments: bool = True):
    
    assert os.path.exists(data_path), errors['Exists data folder']
    def dataset_info(train_dataset, valid_dataset, test_dataset, name):
        
        # Get information from the dataset
        from beautifultable import BeautifulTable as BT
        if hasattr(test_dataset, 'classes'): classes = len(test_dataset.classes)
        elif hasattr(test_dataset, 'labels'): classes = len(np.unique(test_dataset.labels))
        elif hasattr(test_dataset, 'test_labels'): classes = len(np.unique(test_dataset.test_labels))
        else: print('Classes not detected in the dataset', sys.stdout)
        
        # Print information of the dataset
        print('Loading dataset: ', name)
        table = BT()
        table.append_row(['Train Images', len(train_dataset.indices)])
        table.append_row(['Valid Images', len(valid_dataset.indices)])
        table.append_row(['Test Images', len(test_dataset)])
        table.append_row(['Classes', classes])
        print(table)

    root = os.path.join(data_path, dataset)  
    assert os.path.exists(root), errors['Exists particular data folder']      
    
    if dataset == 'CIFAR10':
        transform = transformations(dataset)
        train_dataset = CIFAR10(root = root, download = True, train = True, transform = transform)
        test_dataset  = CIFAR10(root = root, download = False, train = False, transform = transform)
    
    # Append more custom datasets here
    
    # Split training set into traning and validation - normally 90/10 %
    len_ = len(train_dataset)
    train_dataset, valid_dataset = random_split(train_dataset, [round(len_*0.9), round(len_*0.1)])
    
    # Call info of dataset printing function
    if commnets: dataset_info(train_dataset, valid_dataset, test_dataset, name=dataset)
    return train_dataset, valid_dataset, test_dataset