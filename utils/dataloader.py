from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import os
import cv2
from PIL import Image
from path import Path
import tqdm
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision.transforms.functional as tf
import random
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
# from modelsummary import summary
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.open('fail.jpg').load()
from unet import UNet
import warnings
warnings.filterwarnings('ignore')

device = "cuda"

classes = 'building'
bs = 16
class SimDataset(Dataset):
    def __init__(self, count, transform=None,classes = None ,mode = None):
        if mode == 'train':
            self.input_images = Path(os.path.join('data/img')).files()[:-5000]
            self.target_masks = Path(os.path.join(f'data/label/{classes}')).files()[:-5000]
        else:
            self.input_images = Path(os.path.join('data/img')).files()[-5000:]
            self.target_masks = Path(os.path.join(f'data/label/{classes}')).files()[-5000:]
        self.transform = transform
        self.trans_mask = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        
        return len(self.input_images)
    
    def random_rotation(self, image, mask):
        angle = transforms.RandomRotation.get_params([-180, 180])
        image = tf.rotate(image, angle, resample=Image.NEAREST)
        mask = tf.rotate(mask, angle, resample=Image.NEAREST)
        ## 垂直 水平反转     
        if random.random() > 0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        if random.random() > 0.5:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
        return image, mask
    

    def __getitem__(self, idx):
        
        try:
            image = Image.open(self.input_images[idx])
            mask = Image.open(self.target_masks[idx])

            if self.transform:
            #             image = add_noise(image)
                image, mask = random_rotation(image, mask)
                image = self.transform(image)
                mask = self.trans_mask(mask)
                #             print(image.shape,mask.shape)
                #             mask = mask.reshape(256,256)

                return image, mask

        except:
            pass
        
        
       

def add_noise(image):

    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,image.shape[0])
        temp_y = np.random.randint(0,image.shape[1])
        image[temp_x][temp_y] = 255
    return image
    
def random_rotation(image, mask):
    angle = transforms.RandomRotation.get_params([-180, 180])
    image = tf.rotate(image, angle, resample=Image.NEAREST)
    mask = tf.rotate(mask, angle, resample=Image.NEAREST)
    ## 垂直 水平反转     
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    if random.random() > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    return image, mask

# use the same transformations for train/val in this example
trans_train= transforms.Compose([
        transforms.ColorJitter(brightness=1.5, contrast=1.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trans_mask = transforms.Compose([transforms.ToTensor()])


