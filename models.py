import torch
import torch.nn as nn
import torch.nn.functional as F
from args import load_params
from layers import Layer1,Layer2,Layer3,Layer4
import torchvision.models as models

# from efficientnet_pytorch import EfficientNet

# class EfficientNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b0')
#         # 1280 is the number of neurons in last layer. is diff for diff. architecture
#         self.dense_output = nn.Linear(1280, num_classes)

#     def forward(self, x):
#         feat = self.model.extract_features(x)
#         feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
#         return self.dense_output(feat)


class ResNet(nn.Module):
    def __init__(self, num_classes):
       super().__init__()

       self.layers = nn.ModuleList()

       params = load_params()
       if params["channel_mode"] == "fourier" :

        h = params["hidden_dim"]

        self.layers.append(nn.Conv2d(3, 4*h, 8, 8, bias = False))
        self.layers.append(nn.BatchNorm2d(4*h))
        self.layers.append(nn.ReLU)

       self.model_pretrained=models.resnet18(pretrained=True)
       num_features = self.model_pretrained.fc.in_features
       self.model_pretrained.fc = nn.Linear(num_features, num_classes)

       self.layers.append(self.model_pretrained)

    def forward(self,x):
        return self.model_pretrained(x)

class SRNET(nn.Module):

    def __init__(self):
        super(SRNET,self).__init__()
        
        ########################### HYPER PARAMETERS ########################### 
        params = load_params()
        h = params["hidden_dim"]
        num_l2 = params["num_l2"]
        num_l3 = params["num_l3"]
        imsize = params["image_size"] if params["channel_mode"] == "rgb" else int(params["image_size"]/8)
        num_classes = 4 if params["classifier"] == "multi" else 2

        
        ################################# MODEL ################################ 

        self.layers = nn.ModuleList()
        if params["channel_mode"] == "fourier" :
            self.layers.append(nn.Conv2d(3, 4*h, 8, 8, bias = False))
            print("self.layers.append(nn.Conv2d(3,",4*h,"8,8))")
            self.layers.append(nn.BatchNorm2d(4*h))
            print("self.layers.append(nn.BatchNorm2d(",4*h,"))")
            self.layers.append(nn.ReLU())
            print("self.layers.append(nn.ReLU())")
        else: 
            self.layers.append(Layer1(3,4*h))
            print("self.layers.append(Layer1(3,",4*h," ))")

        #two layers of type 1
    
        self.layers.append(Layer1(4*h,h))
        print("self.layers.append(Layer1(",4*h,",",h," ))")
        #five layers of type 2
        for _ in range(num_l2): 
            self.layers.append(Layer2(h,h))
            print("self.layers.append(Layer2(",h,",",h," ))")

        #four layers of type 3
        self.layers.append(Layer3(h,h))
        print("self.layers.append(Layer3(",h,",",h," ))")
        size = 4
        psize = 1
        for _ in range(num_l3 - 1):
            self.layers.append(Layer3(h*psize,h*size))
            print("self.layers.append(Layer3(",h*psize,",",h*size," ))")
            psize = size
            size *= 2
        # self.layers.append(Layer3(h,h))
        # self.layers.append(Layer3(h,4*h))
        # self.layers.append(Layer3(4*h,8*h))
        # self.layers.append(Layer3(8*h,16*h))
        
        #one layer of type 4
        self.layers.append(Layer4(int(h*size/2), imsize, int(imsize/(2**num_l3))))
        print("self.layers.append(Layer4(",h*size/2,",",imsize,",",int(imsize/(2**num_l3))," ))")

        #linear
        self.layers.append(nn.Linear(imsize, num_classes))
        print("self.layers.append(nn.Linear(",imsize,",",num_classes," ))")


    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            # print( x.shape )
        return x 


#for local debugging purposes
class SmallNet(nn.Module) :
    def __init__(self):
        super(SmallNet,self).__init__()

        self.layers = nn.ModuleList()

        #two layers of type 1
        self.layers.append(nn.Conv2d(3,16, 7))

        self.layers.append(nn.MaxPool2d(5))
        self.layers.append(nn.MaxPool2d(5))
        self.layers.append(nn.MaxPool2d(5))


        #linear
        self.lins = nn.ModuleList()
        self.lins.append( nn.Linear(256,4))


    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            # print( x.shape )
        N, C, H, W = x.shape
        x = x.view(N,-1)
        # print("C: " ,C*H*W)
        for layer in self.lins:
            x = layer(x)
        return x
