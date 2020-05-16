import torch
import torch.nn as nn
import torch.nn.functional as F
from args import load_params
from layers import Layer1,Layer2,Layer3,Layer4


#pip install efficientnet-pytorch
# from efficientnet_pytorch import EfficientNet

# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b0')
#         # 1280 is the number of neurons in last layer. is diff for diff. architecture
#         self.dense_output = nn.Linear(1280, num_classes)

#     def forward(self, x):
#         feat = self.model.extract_features(x)
#         feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
#         return self.dense_output(feat)
        

class SRNET(nn.Module):

    def __init__(self):
        super(SRNET,self).__init__()
        
        ########################### HYPER PARAMETERS ########################### 
        params = load_params()
        h = params["hidden_dim"]
        num_l2 = params["num_l2"]
        num_l3 = params["num_l3"]
        imsize = params["image_size"]
        num_classes = 4 if params["classifier"] == "multi" else 2

        
        ################################# MODEL ################################ 

        self.layers = nn.ModuleList()

        #two layers of type 1
        self.layers.append(Layer1(3,4*h))
        self.layers.append(Layer1(4*h,h))
        print("self.layers.append(Layer1(3,",4*h," ))")
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
        self.layers.append(nn.Conv2d(3,16, 5))
        self.layers.append(nn.Conv2d(16,16, 5))
        self.layers.append(nn.MaxPool2d(5))
        self.layers.append(nn.MaxPool2d(5))
        self.layers.append(nn.MaxPool2d(5))
        self.layers.append(nn.MaxPool2d(5))


        #linear

        self.l2 = nn.Linear(16,4)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            # print( x.shape )
        N, C, H, W = x.shape
        # print("C: " ,C*H*W)
        return self.l2(x.view(N,-1))
