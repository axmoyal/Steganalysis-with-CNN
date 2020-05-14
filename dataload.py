"""
Description: This file contains the dataloader class and all data processing methods. 

Author: Lucas Soffer, Guillermo Bezos, Axel Moyal
"""

import torch
from torch.utils.data import Dataset, DataLoader
import imageio

NUM_IM_PER_FILE = 8005 #Number of images per file
NUM_TEST_IM = 5000 #Number of test images

class Alaska(Dataset): 
    """
    Description: Alaska dataset loader.
    More information can be found at:
    https://www.kaggle.com/c/alaska2-image-steganalysis/data?select=Cover

    Args: 
        dirpath (str) : Path to directory containing dataset.   
        Example: Lucas's path is C:/Users/lucas/Documents/Stanford/CS231n/DNN_Steganalysis/data
    """

    def __init__(self, dirpath):
        self.path = dirpath

    """ 
    Description: returns size of dataset
    """
    def __len__(self) : 
        return  4*NUM_IM_PER_FILE

    """ 
    Description : Pulls image from dataset
    Usage: dataloader[idx] 
    Args: 
        idx: index of image. Images are indexed as: 
            Cover: 1 to 8005            Label 0
            JMiPOD: 8006 to 16010       Label 1
            JUNIWARD: 16011 to 24015    Label 2
            UERD: 24016 to 32020        Label 3
    Returns: 
        datapoint (tuple) : (image (tensor), label (int))
    """
    def __getitem__(self, idx): 
        ind = idx % NUM_IM_PER_FILE
        if (idx <= NUM_IM_PER_FILE):
            image = imageio.imread(self.path + "/Cover/" + str(ind).rjust(5, '0') + ".jpg")
            label = 0
        elif (idx <= 2*NUM_IM_PER_FILE):
            image = imageio.imread(self.path + "/JMiPOD/" + str(ind).rjust(5, '0') + ".jpg")
            label = 1
        elif (idx <= 3*NUM_IM_PER_FILE): 
            image = imageio.imread(self.path + "/JUNIWARD/" + str(ind).rjust(5, '0') + ".jpg")
            label = 2
        elif (idx <= 4*NUM_IM_PER_FILE)
            image = imageio.imread(self.path + "/UERD/" + str(ind).rjust(5, '0') + ".jpg") 
            label = 4
        else: 
            raise IndexError("Index out of range!")
        return (torch.tensor(image), label)


class AlaskaTest(Dataset): 
    """
    Description: Alaska dataset test loader
    Args: 
        dirpath (str) : Path to directory containing dataset.   
        Example: Lucas's path is C:/Users/lucas/Documents/Stanford/CS231n/DNN_Steganalysis/data
    """

    def __init__(self, dirpath):
        self.path = dirpath

    """ 
    Description: returns size of dataset
    """
    def __len__(self) : 
        return NUM_TEST_IM

    """ 
    Description : Pulls image from dataset
    Usage: dataloader[idx] 
    Args: 
        idx: index of test image (1 to 5000)
    Returns: 
        datapoint: image
    """
    def __getitem__(self, idx): 
        return imageio.imread(self.path + "/Test/" + str(idx).rjust(4, '0') + ".jpg")
        raise IndexError("Index out of range!")
