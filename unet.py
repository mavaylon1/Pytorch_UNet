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
        #print("First Conv Layer",x1.size())
        x2=self.max_pool_2x2(x1)
        x3=self.down_conv_2(x2) #
        #print("Second",x3.size())
        x4=self.max_pool_2x2(x3)
        x5=self.down_conv_3(x4) #
        #print("third", x5.size())


        x6=self.max_pool_2x2(x5)
        x7=self.down_conv_4(x6)
        #print("x7",x7.size()) #
        x8=self.max_pool_2x2(x7)
        x9=self.down_conv_5(x8)

        #decoder
        x= self.up_trans_1(x9)
        #print("x to bo concat with x7",x.size())
        x =self.up_conv_1(torch.cat([x,x7],1))
        #print("x7 concat",x.size())

        x= self.up_trans_2(x)

        x =self.up_conv_2(torch.cat([x,x5],1))

        x= self.up_trans_3(x)

        x =self.up_conv_3(torch.cat([x,x3],1))

        x= self.up_trans_4(x)

        x =self.up_conv_4(torch.cat([x,x1],1))

        x=self.out(x)

        #print("output", x.size())
        return x
