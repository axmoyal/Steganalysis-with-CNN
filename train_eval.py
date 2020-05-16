
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from dataload import Alaska
from models import SRNET
from utils import get_available_devices


# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)

#     def forward(self, x):
#         x = x.reshape(N,)
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred

def get_dataloaders(alaska_dataset,b_size,frac_test=0.005):
	N=len(alaska_dataset)
	lengths = [int(N*(1-frac_test)), int(N*frac_test)]
	train_set, dev_set = td.random_split(alaska_dataset, lengths)
	train_loader=td.DataLoader(train_set, batch_size=b_size)
	dev_loader=td.DataLoader(dev_set, batch_size=b_size)
	return train_loader,dev_loader

def prepbatch(X, y) : 
    X = X.view(-1, 3, 512, 512)
    y = y.view(-1)
    return X, y

        
# train the model from a dataloader and evaluate it every 5 batch  on the dev dataloader
def train(train_loader,dev_loader,model, device, num_batch=0,path=None,lear_rate=1e-4,N_epoch=10):
    tb_writer = SummaryWriter()
    opti= torch.optim.Adam(model.parameters())
    for epoch in range(N_epoch):
        for batch_index,(X,y_label) in enumerate(train_loader):


            X = X.to(device)
            y_label = y_label.to(device)

            X, y_label = prepbatch(X, y_label)



            opti.zero_grad()

            y_pred=model(X)
           

            loss=F.cross_entropy(y_pred,y_label)           
            loss_value=loss.item()
            print('Batch loss: {}'.format(loss))
            loss.backward()
            opti.step()  
            tb_writer.add_scalar('batch train loss', loss_value, epoch*num_batch+batch_index)
            if batch_index%5==0:
                loss_dev,accuracy_dev=eval_model(model,dev_loader, device)
                print('Dev Loss: {}'.format(loss_dev))
                print('Accuracy: {}'.format(accuracy_dev))
                tb_writer.add_scalar('dev loss', loss_dev, epoch*num_batch+batch_index)
                tb_writer.add_scalar('dev accuracy', accuracy_dev, epoch*num_batch+batch_index)
            #torch.save(model.state_dict(), path)

# evaluate the model on a loader.
def eval_model(model,loader, device):
    model.eval()
    LOSS=0
    accuracy=0
    num = 0
    for batch_index,(X,y_label) in enumerate(loader):

        X = X.to(device)
        y_label = y_label.to(device)

        X, y_label = prepbatch(X, y_label)
        num += X.shape[0]
        y_pred=model(X)
        loss=F.cross_entropy(y_pred,y_label)
        LOSS+=loss.item()
        #accuracy+=(y_label.eq(y_pred.long()).sum()
        _, pred_classes = y_pred.max(axis = 1)
        accuracy+=y_label.eq(pred_classes.long()).sum()
        # print("Eval successful")
    print(accuracy)
    accuracy=accuracy.item()/num
    LOSS=LOSS/num
    return LOSS,accuracy
    

def test(test_loader,Model,path):
	model = Model
	model.load_state_dict(torch.load(path))
	loss,accuracy=eval_model(model,test_loader)
	print('Test Loss : '+str(loss))
	print('Test Accuract : '+str(accuracy))

if __name__ == '__main__':
    device, gpu_ids = get_available_devices()
    print(device)
    AlaskaDataset=Alaska("./data","pairs",1, "multi")
    model = SRNET()  
    model = model.to(device)
    train_loader,dev_loader=get_dataloaders(AlaskaDataset,2)
    train(train_loader,dev_loader,model, device)
