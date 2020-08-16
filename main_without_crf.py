import torch
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from unet import UNet

from skimage import data
from skimage.color import rgb2gray

label_path="/Users/mavaylon/Research/Research_Gambier/Data/Correct_Labels/"
data_path="/Users/mavaylon/Research/Research_Gambier/Data/gamb_orig/"

model=UNet()
device=torch.device("cpu")
model.to(device)
optimizer= torch.optim.Adam(model.parameters(),.001)
criterion= nn.CrossEntropyLoss()

m = nn.Sigmoid()
loss_ = nn.BCELoss()

names_data=os.listdir(data_path)
names_data.sort()
names_labels= os.listdir(label_path)
names_labels.sort()
gt_list = [cv2.imread(label_path+name,0) for name in names_labels]
data_list= [cv2.imread(data_path+name,0) for name in names_data]

gt_list= [gt_list[0]]
data_list=[data_list[0]]

for epoch in range(10):
    for x,gt in zip(data_list,gt_list):
        x=x/np.max(x)
        gt=(gt>0).astype(int)
        x = np.reshape(x,[1,1,x.shape[0],x.shape[1]])
        gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1]])
        x = torch.from_numpy(x).float()
        gt=torch.from_numpy(gt).long()
        optimizer.zero_grad()
        y=model(x)
        loss = criterion(y,gt)
        loss.backward()
        optimizer.step()
        #print(y.detach().numpy()[0,:,:,:].shape)
        y=y.detach().numpy()[0,0,:,:]
        #gray=rgb2gray(y)
        #plt.figure()
        #plt.imshow(y)
        #plt.show()
        #print(y.shape)
        print(y*255)
        print(loss)
    cv2.imwrite(str(epoch)+".png",y*255)
