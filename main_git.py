import torch
import torch.nn.functional as F
from unet_git import UNet
import torch.nn as nn

model = UNet(n_classes=2, padding=True, up_mode='upsample')
optim = torch.optim.Adam(model.parameters())
epochs = 1
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
m = nn.LogSoftmax(dim=1)
loss_ = nn.NLLLoss()
label_path="/Users/mavaylon/Research/Research_Gambier/Data/Correct_Labels/"
data_path="/Users/mavaylon/Research/Research_Gambier/Data/gamb_orig/"

names_data=os.listdir(data_path)
names_data.sort()
names_labels= os.listdir(label_path)
names_labels.sort()
gt_list = [cv2.imread(label_path+name,0) for name in names_labels]
data_list= [cv2.imread(data_path+name,0) for name in names_data]

for epoch in range(1):
    for x,gt in zip(data_list,gt_list):
        x=x/np.max(x)
        gt=(gt>0).astype(int)
        x = np.reshape(x,[1,1,x.shape[0],x.shape[1]])
        gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1]])
        x = torch.from_numpy(x).float()
        gt=torch.from_numpy(gt).long()
        optim.zero_grad()
        y=model(x)
        loss = loss_(m(y),gt)
        loss.backward()
        optim.step()
        print(loss)
    print(loss)
