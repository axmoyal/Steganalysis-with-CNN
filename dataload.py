"""
Description: This file contains the dataloader class and all data processing methods. 

Author: Lucas Soffer, Guillermo Bezos, Axel Moyal
"""

import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
from args import load_params
from utils import dct2

NUM_IM_PER_FILE = 75000 #Number of images per file
NUM_TEST_IM = 5000 #Number of test images
NUM_FILES = 4 #Number of files
params = load_params()



############################## CLASSES ###############################
class Alaska(Dataset): 
    """
    Description: Alaska dataset loader.
    More information can be found at:
    https://www.kaggle.com/c/alaska2-image-steganalysis/data?select=Cover

    Args: 
        dirpath (str) : Path to directory containing dataset. 
        mode (str): Indicates the format return format for training:
            "single" : Single image
            "pairs" : Image pairs (corrupted, non-corrupted)
            "quads" : Image quads (covert, JMiPOD, JUNIWARD, UERD)
        scale (int): Value between 1 and 8 which scales the dataset through data augmentation: 
            1 - 4 : creates rotated images (original, rot90, rot180, rot270)
            5 - 8 : creates flipped and rotate images (fliplr, fliplr rot90, ...)
            Example: Selecting 5 would multiply the amount of data by 5 by including (original, rot90, rot180, rot270, fliplr)
        classifier (str) : 
            "multi" : Multi class
            "binary" : Binary class
    """

    def __init__(self):
        ##################### GET PARAMS ######################
        mode = params["mode"]
        scale = params["scale"]
        classifier = params["classifier"]
        dirpath = "./data"

        ######################## INIT #########################
        self.path = dirpath
        self.scale = scale
        self.isbinary = True if classifier == "binary" else False
        
        #Initiate mode specific data
        if mode == "single" :
            self.mode = 0 
            self.size = NUM_IM_PER_FILE * NUM_FILES * self.scale
            self.getdatapoint = single
        elif mode == "pairs": 
            self.mode = 1
            self.size = NUM_IM_PER_FILE * (NUM_FILES - 1) * self.scale
            self.getdatapoint = pairs
        elif mode == "quads":
            self.mode = 2
            self.size = NUM_IM_PER_FILE * self.scale
            self.getdatapoint = quads
        else : 
            raise ValueError ( mode + " is not a valid mode")

    """ 
    Description: returns size of dataset. 
    """
    def __len__(self) :
        return self.size

    """ 
    Description : Pulls image from dataset
    Usage: dataloader[idx] 
    Args: 
        idx: index into dataset
    Returns: 
        datapoint (tuple) : depends on mode
    """

    def __getitem__(self, idx): 
        return self.getdatapoint(idx, self.scale, self.path, self.isbinary)

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
        return torch.tensor(imageio.imread(self.path + "/Test/" + str(idx).rjust(5, '0') + ".jpg"), dtype= torch.float).permute(2,0,1)


################################# HELPER FUNCTIONS ###################################
""" 
Description : Pulls image from dataset in single mode
Args: 
    idx: index into dataset
    scale: data augmentation scale
    path: Path to dataset
    isbinary: bool indicating if it is a binary classification 
Returns: 
    datapoint (tuple) : (image, class)
"""
def single(idx, scale, path, isbinary) :
    ind = list(np.unravel_index(idx, (NUM_IM_PER_FILE, NUM_FILES, scale)))
    ind[0] += 1
    label = 1 if isbinary else ind[1] 
    if (ind[1] == 0):
        image = imageio.imread(path + "/Cover/" + str(ind[0] ).rjust(5, '0') + ".jpg")
        label = 0
    elif (ind[1] == 1):
        image = imageio.imread(path + "/JMiPOD/" + str(ind[0] ).rjust(5, '0') + ".jpg")
    elif (ind[1] == 2): 
        image = imageio.imread(path + "/JUNIWARD/" + str(ind[0]).rjust(5, '0') + ".jpg")
    elif (ind[1] == 3):
        image = imageio.imread(path + "/UERD/" + str(ind[0]).rjust(5, '0') + ".jpg") 
    return (transform(image, ind[2]), label)


""" 
Description : Pulls image from dataset in pairs mode
Args: 
    idx: index into dataset
    scale: data augmentation scale
    path: Path to dataset
    isbinary: bool indicating if it is a binary classification 
Returns: 
    datapoint (tuple) : (tuple (original, corrupted), label(og label, corrupted label))
"""
def pairs(idx, scale, path, isbinary):
    ind = list(np.unravel_index(idx, (NUM_IM_PER_FILE, NUM_FILES - 1, scale)))
    label = 1 if isbinary else ind[1] + 1 
    og = imageio.imread(path + "/Cover/" + str(ind[0]).rjust(5, '0') + ".jpg")
    if (ind[1] == 0):
        image = imageio.imread(path + "/JMiPOD/" + str(ind[0]).rjust(5, '0') + ".jpg")
    elif (ind[1] == 1): 
        image = imageio.imread(path + "/JUNIWARD/" + str(ind[0]).rjust(5, '0') + ".jpg")
    elif (ind[1] == 2):
        image = imageio.imread(path + "/UERD/" + str(ind[0]).rjust(5, '0') + ".jpg") 
    return(torch.stack((transform(og, ind[2]), transform(image, ind[2])), dim = 0), torch.tensor((0, label)))

""" 
Description : Pulls image from dataset in quads mode
Args: 
    idx: index into dataset
    scale: data augmentation scale
    path: Path to dataset
    isbinary: bool indicating if it is a binary classification 
Returns: 
    datapoint (tuple) : (tuple(original, JMiPOD, JUNIWARD, UERD) ,tuple(0, c1, c2 ,c3))
"""
def quads(idx, scale, path, isbinary):
    ind = list(np.unravel_index(idx, (NUM_IM_PER_FILE, scale)))
    labels = [1, 1, 1] if isbinary else [1 , 2, 3]
    og = transform(imageio.imread(path + "/Cover/" + str(ind[0]).rjust(5, '0') + ".jpg"), ind[1])
    im1 = transform(imageio.imread(path + "/JMiPOD/" + str(ind[0]).rjust(5, '0') + ".jpg"), ind[1])
    im2 = transform(imageio.imread(path + "/JUNIWARD/" + str(ind[0]).rjust(5, '0') + ".jpg"), ind[1])
    im3 = transform(imageio.imread(path + "/UERD/" + str(ind[0]).rjust(5, '0') + ".jpg"), ind[1])
    return (torch.stack((og, im1, im2, im3), dim = 0), torch.tensor((0, labels[0], labels[1], labels[2])))


"""
Description : Transforms image
Args: 
    image: orginal image
    transformation: number between 1 and 8 to decide transformation
"""
def transform(image, transformation): 
    if params["channel_mode"] == "fourier":
        image = dct2(image)
    elif params["channel_mode"] == "rgb": 
        image = torch.tensor(image, dtype= torch.float).permute(2,0,1)
    else:
        raise ValueError("Provide a valid channel_mode, [fourier/rgb]")

    if transformation == 0: return image
    if transformation == 1: return image.rot90(1, [1, 2])
    if transformation == 2: return image.rot90(2, [1, 2])
    if transformation == 3: return image.rot90(3, [1, 2])
    if transformation == 4: return image.flip(2)
    if transformation == 5: return image.rot90(1, [1, 2]).flip(2)
    if transformation == 6: return image.rot90(2, [1, 2]).flip(2)
    if transformation == 7: return image.rot90(3, [1, 2]).flip(2)
    raise ValueError ("Not a valid transformation")



########################### TEST ################################

# import matplotlib.pyplot as plt
# d1 = Alaska("C:/Users/lucas/Documents/Stanford/CS231n/DNN_Steganalysis/data", "single", 1, "binary")
# d2 = Alaska("C:/Users/lucas/Documents/Stanford/CS231n/DNN_Steganalysis/data", "pairs", 8, "binary")
# d3 = Alaska("C:/Users/lucas/Documents/Stanford/CS231n/DNN_Steganalysis/data", "quads", 5, "multi")
# t1 = AlaskaTest("C:/Users/lucas/Documents/Stanford/CS231n/DNN_Steganalysis/data")
# while True: 
#     ind1 = np.random.randint(0, len(d1))
#     ind2 = np.random.randint(0, len(d2))
#     ind3 = np.random.randint(0, len(d3))
#     ind4 = np.random.randint(0, len(t1))
#     plt.imshow(d1[ind1][0].permute(1,2,0))
#     plt.title("Label d1 " + str(d1[ind1][1]))
#     plt.show()
#     for i in [0,1]: 
#         plt.imshow(d2[ind2][0][i].permute(1,2,0))
#         plt.title("Label d2 " + str(d2[ind2][1][i]))
#         plt.show()
#     for i in [0,1,2,3]: 
#         plt.imshow(d3[ind3][0][i].permute(1,2,0))
#         plt.title("Label d3 " + str(d3[ind3][1][i]))
#         plt.show()
#     plt.imshow(t1[ind4].permute(1,2,0))
#     plt.title("Label t1" )
#     plt.show()
