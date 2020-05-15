import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Layer1,Layer2,Layer3,Layer4

class SRNET(nn.Module):

    def __init__(self):
        super(SRNET,self).__init__()

        self.layers = nn.ModuleList()

        #two layers of type 1
        self.layers.append(Layer1(3,64))
        self.layers.append(Layer1(64,16))

        #five layers of type 2
        self.layers.append(Layer2(16,16))
        self.layers.append(Layer2(16,16))
        self.layers.append(Layer2(16,16))
        self.layers.append(Layer2(16,16))
        self.layers.append(Layer2(16,16))

        #four layers of type 3
        self.layers.append(Layer3(16,16))
        self.layers.append(Layer3(16,64))
        self.layers.append(Layer3(64,128))
        self.layers.append(Layer3(128,256))
        
        #one layer of type 4
        self.layers.append(Layer4(256,512,32))

        #linear
        self.layers.append(nn.Linear(512,4))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            print( x.shape )
        return x 

