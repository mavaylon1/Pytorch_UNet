import cv2
import matplotlib.pyplot as plt

import glob
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np

def crop_center(img, cropx, cropy):
       y, x, z = img.shape
       startx = x // 2 - (cropx // 2)
       starty = y // 2 - (cropy // 2)
       return img[starty:starty + cropy, startx:startx + cropx]

output_path = "/Users/mavaylon/Research/Pytorch_UNet/Data/Talita/Talita_500x500/orig/"
out2="/Users/mavaylon/Research/Pytorch_UNet/Data/Talita/Talita_500x500/gt/"
img_names = sorted(glob.glob("/Users/mavaylon/Research/Pytorch_UNet/Data/Talita/gamb_orig/*.png"))
img_names2 = sorted(glob.glob("/Users/mavaylon/Research/Pytorch_UNet/Data/Talita/Correct_Labels/*.png"))

#img=cv2.imread(img_names[0])
#q,w,e=img.shape

#print(img_names[0].split("/")[-1])

for name in img_names:
    print(name)
    im = cv2.imread(name)
    print(im.shape)
    im = crop_center(im,500,500)
    print(im.shape)
    out_name = name.split("/")[-1]
    imsave(output_path+out_name,im)

for name in img_names2:
    print(name)
    im = cv2.imread(name)
    print(im.shape)
    im = crop_center(im,500,500)
    print(im.shape)
    out_name = name.split("/")[-1]
    imsave(out2+out_name,im)
