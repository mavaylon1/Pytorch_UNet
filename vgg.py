import torch
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#from vgg_crf_setup import CrfRnnNet
from vgg_git import VGG_net
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model= VGG_net().to(dev)
#model= VGG_net(in_channels=3, num_classes=2).to(dev)
optimizer= torch.optim.Adam(model.parameters(),.001)
criterion= nn.CrossEntropyLoss()

label_path="/Users/mavaylon/Research/Pytorch_UNet/500x500/orig/"
data_path="/Users/mavaylon/Research/Pytorch_UNet/500x500/gt/"

names_data=os.listdir(data_path)
names_data.sort()
names_labels= os.listdir(label_path)
names_labels.sort()
gt_list = [cv2.imread(label_path+name) for name in names_labels]
data_list= [cv2.imread(data_path+name) for name in names_data]

gt_list= [gt_list[0]]
data_list=[data_list[0]]

#print(gt_list[0].shape)

for epoch in range(100):
    for x,gt in zip(data_list,gt_list):
        x=x/np.max(x)
        gt=(gt>0).astype(int)
        x_=np.copy(x)
        gt_=np.copy(gt)
        x = np.reshape(x,[1,3,x.shape[0],x.shape[1]])
        gt = np.reshape(gt,[3,gt.shape[0],gt.shape[1]])
        x = torch.from_numpy(x).float().to(dev)
        gt=torch.from_numpy(gt).long().to(dev)
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y,gt)
        loss.backward()
        optimizer.step()

        #print(y)


        y_ = y[0].squeeze().cpu()
        y_ = torch.argmax(y_,dim=0).detach().numpy()

        print(loss)
    out = np.hstack([x_*255,gt_*255,y_*255])
    cv2.imwrite('/Users/mavaylon/Research/Pytorch_UNet/CRF_Images_500x500/'+str(epoch)+".png",out)
