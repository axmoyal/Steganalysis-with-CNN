import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer1(nn.Module):
    """ Layer of type 1 
    Args:
        in_channels (int) : number of in channels 
        out_channels (int) : number of out channels
        kernel_size (int) : size of kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = 1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.batch_norm(self.conv(x)))


class Layer2(nn.Module):
    """ Layer of type 2 
    Args:
        in_channels (int) : number of in channels 
        out_channels (int) : number of out channels
        kernel_size (int) : size of kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        self.l1 = Layer1(in_channels, out_channels, kernel_size)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding = 1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)

        
    def forward(self, x):
        return x + self.batch_norm(self.conv(self.l1(x)))

class Layer3(nn.Module):
    """ Layer of type 3
    Args:
        in_channels (int) : number of in channels 
        out_channels (int) : number of out channels
        kernel_size (int) : size of kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3):


        self.conv_hwy = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 2)
        self.batch_norm_hwy =  torch.nn.BatchNorm2d(out_channels)

        self.l1 = Layer1(in_channels, out_channels, kernel_size)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding = 1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.av_pool = nn.AvgPool2d(3, stride = 2)

        
    def forward(self, x):
        return self.batch_norm_hwy(self.conv_hwy(x)) + self.av_pool(self.batch_norm(self.conv(self.l1(x))))


class Layer4(nn.Module):
    """ Layer of type 4
    Args:
        in_channels (int) : number of in channels 
        out_channels (int) : number of out channels
        kernel_size (int) : size of kernel
    """

    def __init__(self, in_channels, out_channels, kernel_size = 3):
        self.l1 = Layer1(in_channels, out_channels, kernel_size)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding = 1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.glob_pool = nn.AvgPool2d(16)

    def forward(self,x):
        N, C, H, W = x.shape
        return self.glob_pool(self.batch_norm(self.conv(self.l1(x)))).view((N,C))






