
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import sys
import pandas as pd
from tqdm import tqdm


#from torch.utils.tensorboard import SummaryWriter
from dataload import AlaskaTest
from models import *
from utils import get_available_devices, AverageMeter, alaska_weighted_auc
from args import *


def test(test_loader, model, device):
    total_predictions = [] 
    with torch.no_grad():
        for X in tqdm(test_loader):
            X = X.to(device)
            y_pred = model(X)
            scores = F.softmax(y_pred, dim = 1)
            total_predictions.append(np.array(scores.cpu()))

    total_predictions = np.concatenate(total_predictions, axis = 0)
    scores = multi_to_binary(total_predictions)

    print(len(np.arange(1,5001)))
    print(len(scores))
    df = pd.DataFrame({'Id':np.arange(1,5001), 'Label':scores})
    df['Id'].apply(lambda x : f'{x:04}')
    df.to_csv("save/" + name + "/" +name +".csv")

def multi_to_binary(y_pred):
    temp = np.maximum(y_pred[:,1],y_pred[:,2],y_pred[:,3])
    scores = temp / (y_pred[:,0] + temp)
    return scores 

if __name__ == '__main__':


    name = sys.argv[1]

    params = load_params("save/" + name + "/" + name +".json")

    device, gpu_ids = get_available_devices()
    print(device)

    AlaskaDataset= AlaskaTest("data/")
    #model = Net(4)
    #model =SmallNet()
    #model = SRNET()
    model = ResNet(4)
    model.load_state_dict(torch.load("save/" + name + "/" + name +".pkl"))
    model = model.to(device)

    model.eval()

    test_loader = td.DataLoader(AlaskaDataset,  batch_size=params["batch_size"], shuffle = False)

    test(test_loader,model, device)






