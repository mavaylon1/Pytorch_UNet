import torch
import torch.nn as nn
from unetcrf import Unetcrfnet
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
label_path="/Users/mavaylon/Research/Research_Gambier/Data/Correct_Labels/"
data_path="/Users/mavaylon/Research/Research_Gambier/Data/gamb_orig/"

model= Unetcrfnet()
device=torch.device("cpu")
model.to(device)
optimizer= torch.optim.Adam(model.parameters(),.001)
criterion= nn.CrossEntropyLoss()


names_data=os.listdir(data_path)
names_data.sort()
names_labels= os.listdir(label_path)
names_labels.sort()
gt_list = [cv2.imread(label_path+name,0) for name in names_labels]
data_list= [cv2.imread(data_path+name,0) for name in names_data]
#gt_list= [gt_list[0]]
#data_list=[data_list[0]]
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(gt[256])
    #plt.subplot(1,2,2)
    #plt.imshow(data[256])
    #plt.show()
for epoch in range(1):
    for x,gt in zip(data_list,gt_list):
        x=x/np.max(x)
        gt=(gt>0).astype(int)
        x = np.reshape(x,[1,1,x.shape[0],x.shape[1]])
        gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1]])
        x = torch.from_numpy(x).float()
        gt=torch.from_numpy(gt).long()
        optimizer.zero_grad()
        y=model(x)
        #y=torch.log(y+1e-20)
        #print(y)
        loss = criterion(y,gt)
        loss.backward()
        optimizer.step()
        print(loss)
    #print(loss)
