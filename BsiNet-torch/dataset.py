"""
The role of this file completes the data reading
"dist_mask" is obtained by using Euclidean distance transformation on the mask
"dist_contour" is obtained by using quasi-Euclidean distance transformation on the mask
"""


import numpy as np
import cv2
from PIL import Image, ImageFile

from skimage import io
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import os
from glob import glob
from torch.utils.data import DataLoader
import rasterio
import matplotlib.pyplot as plt
import torch

class Ai4smallDataset():
    def __init__(self,image_list,flag,dict_format=True):

        self.image_list = image_list
        self.image_ids = [x.split("/")[-1].split(".")[0] for x in self.image_list]
        self.parcel_list= [f"../original/sentinel-2-asia/parcel_mask/{x}.tif" for x in self.image_ids]
        self.mask_list= [f"../original/sentinel-2-asia/{flag}/masks/{x}.tif" for x in self.image_ids]
        self.dict_format=dict_format

    def __getitem__(self, item):

        image_path = self.image_list[item]
        #image_path = os.path.join(self.image_path, image_name)
        input_image = self.__open_tiff__(image_path)
        input_image = input_image[:3,:,:]
        input_image = self.min_max_normalize(input_image)
        input_image = self.resize_array(input_image,(512,512))
        #input_image = image.resize((1024, 1024), Image.ANTIALIAS)
        input_image = torch.tensor(input_image).float()

        parcel_path = self.parcel_list[item]
        parcel_label = self.__open_tiff__(parcel_path)
        parcel_label = self.resize_array(parcel_label,(512,512),mask=True)
        parcel_label = torch.tensor(parcel_label)
    
        mask_path = self.mask_list[item]
        #label_path = os.path.join(self.parcel_path, label_name)
        mask_label = self.__open_tiff__(mask_path)
        mask_label = self.resize_array(mask_label,(512,512),mask=True)
        mask_label = torch.tensor(mask_label)
        return input_image,mask_label,parcel_label

    def __len__(self):

        return len(self.image_list)
    
    def min_max_normalize(self,image):
        # Assuming image shape is n_channels x height x width
        channel_mins=np.min(image,axis=(1,2,),keepdims=True)
        channel_maxs=np.max(image,axis=(1,2),keepdims=True) 
        normalized_array = (image-channel_mins)/(channel_maxs-channel_mins)
        return normalized_array

    def __open_tiff__(self,image_path):
        image=rasterio.open(image_path).read()
        return image
        
    def resize_array(self,array, new_size,mask=False):
        # Transpose the array to height x width x channels
        array = np.transpose(array, (1, 2, 0))
        if mask:
            array=np.squeeze(array,2)
            

        array = array * 255
        # Convert to PIL Image
        pil_image = Image.fromarray(array.astype(np.uint8))

        # Resize the image
        resized_image = pil_image.resize(new_size)

        # Convert back to numpy array
        resized_array = np.array(resized_image)
        if mask:
            resized_array=np.expand_dims(resized_array,2)

        # Transpose back to channels x height x width
        resized_array = np.transpose(resized_array, (2, 0, 1))
        
        resized_array = resized_array/255

        return resized_array