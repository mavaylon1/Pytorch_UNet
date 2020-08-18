import torch
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from unet import UNet



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=UNet().to(dev)
optimizer= torch.optim.Adam(model.parameters(),.001)


criterion= nn.CrossEntropyLoss()
label_path="../Research_Gambier/Data/Correct_Labels/"
data_path="../Research_Gambier/Data/gamb_orig/"

names_data=os.listdir(data_path)
names_data.sort()
names_labels= os.listdir(label_path)
names_labels.sort()
gt_list = [cv2.imread(label_path+name,0) for name in names_labels]
data_list= [cv2.imread(data_path+name,0) for name in names_data]

gt_list= [gt_list[0]]
data_list=[data_list[0]]

for epoch in range(100):
    for x,gt in zip(data_list,gt_list):
        x=x/np.max(x)
        gt=(gt>0).astype(int)
        x_=np.copy(x)
        gt_=np.copy(gt)
        x = np.reshape(x,[1,1,x.shape[0],x.shape[1]])
        gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1]])
        x = torch.from_numpy(x).float().to(dev)
        gt=torch.from_numpy(gt).long().to(dev)
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y,gt)
        loss.backward()
        optimizer.step()

        print(y)


        y_ = y[0].squeeze().cpu()
        y_ = torch.argmax(y_,dim=0).detach().numpy()

        print(loss)
    out = np.hstack([x_*255,gt_*255,y_*255])
    cv2.imwrite('/Users/mavaylon/Research/Pytorch_UNet/Base_Unet_images/'+str(epoch)+".png",out)
