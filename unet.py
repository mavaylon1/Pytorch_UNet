import torch
import torch.nn as nn

def double_conv(in_c,out_c):
    conv = nn.Sequential(
    nn.Conv2d(in_c,out_c,kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_c,out_c,kernel_size=3, padding=1),
    nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size=target_tensor.size()[2]
    tensor_size=tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    output_tensor=(tensor[:,:,delta:tensor_size-delta, delta:tensor_size-delta])
    print(output_tensor.shape, target_tensor.shape)
    return output_tensor

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.up_conv_1 = double_conv(1024,512)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.up_conv_2 = double_conv(512,256)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.up_conv_3 = double_conv(256,128)
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.up_conv_4 = double_conv(128,64)

        self.out= nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1)


    def forward(self,image):
        #bs, c, h, w
        #encoder
        x1=self.down_conv_1(image) #
        x2=self.max_pool_2x2(x1)
        x3=self.down_conv_2(x2) #
        x4=self.max_pool_2x2(x3)
        x5=self.down_conv_3(x4) #
        x6=self.max_pool_2x2(x5)
        x7=self.down_conv_4(x6) #
        x8=self.max_pool_2x2(x7)
        x9=self.down_conv_5(x8)

        #decoder
        x= self.up_trans_1(x9)
        x =self.up_conv_1(torch.cat([x,x7],1))

        x= self.up_trans_2(x)

        x =self.up_conv_2(torch.cat([x,x5],1))

        x= self.up_trans_3(x)

        x =self.up_conv_3(torch.cat([x,x3],1))

        x= self.up_trans_4(x)

        x =self.up_conv_4(torch.cat([x,x1],1))

        x=self.out(x)
        return x





if __name__ == '__main__':
    #image = torch.rand((1,1,572,572))
    model= UNet()
    device=torch.device("cpu")
    model.to(device)
    optimizer= torch.optim.Adam(model.parameters(),.001)
    criterion= nn.CrossEntropyLoss()
    #print(model.forward(image))
    import os
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    label_path="/Users/matthewavaylon/Research_Gambier/Research_Gambier/Data/Correct_Labels/"
    data_path="/Users/matthewavaylon/Research_Gambier/Research_Gambier/Data/gamb_orig/"

    names_data=os.listdir(data_path)
    names_data.sort()
    names_labels= os.listdir(label_path)
    names_labels.sort()
    gt_list = [cv2.imread(label_path+name,0) for name in names_labels]
    data_list= [cv2.imread(data_path+name,0) for name in names_data]
    gt_list= [gt_list[0]]
    data_list=[data_list[0]]
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(gt[256])
    #plt.subplot(1,2,2)
    #plt.imshow(data[256])
    #plt.show()
    for epoch in range(100):
        for x,gt in zip(data_list,gt_list):
            x=x/np.max(x)
            gt=(gt>0).astype(int)
            x = np.reshape(x,[1,1,x.shape[0],x.shape[1]])
            gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1]])
            x = torch.from_numpy(x).float()
            gt=torch.from_numpy(gt).long()
            gt = gt.to(device)
            x=x.to(device)
            optimizer.zero_grad()
            y=model(x)
            loss = criterion(y,gt)
            loss.backward()
            optimizer.step()
            print(loss.item())
