import csv
import os
#import cv2
from copy import deepcopy
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
import random
import rasterio
class TrainDataset():
    def __init__(self, image_id, label_id, is_robustness,dict_format=True):

        self.image_id = image_id
        self.label_id = label_id
        self.is_robustness = is_robustness
        self.dict_format=dict_format

        self.image_list = [f"./ai4boundaries_data/orthophoto/images/train/{x}_ortho_1m_512.tif" for x in self.image_id]
        self.label_list = [f"./ai4boundaries_data/orthophoto/masks/train/{x}_ortholabel_1m_512.tif" for x in self.image_id]

        if self.is_robustness:
            self.image_list, self.label_list = self.get_images_and_labels_path_for_loop()

    def __getitem__(self, item):

        image_path = self.image_list[item]
        #image_path = os.path.join(self.image_path, image_name)
        input_image = self.__open_tiff__(image_path)
        input_image = self.min_max_normalize(input_image)
        input_image = self.resize_array(input_image,(1024,1024))
        #input_image = image.resize((1024, 1024), Image.ANTIALIAS)
        input_image = torch.tensor(input_image).float()

        label_path = self.label_list[item]
        #label_path = os.path.join(self.label_path, label_name)
        input_label = self.__open_tiff__(label_path)
        input_label = self.resize_array(input_label,(256,256))
        
        #label = label.resize((256, 256), Image.ANTIALIAS)

        input_label = torch.tensor(input_label).long()

        points_scale = np.array(input_image.shape[1:])[None, ::-1]
        point_grids = build_all_layer_point_grids(
            n_per_side=32,
            n_layers=0,
            scale_per_layer=1,
        )
        points_for_image = point_grids[0] * points_scale
        in_points = torch.as_tensor(points_for_image)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int)
        points = (in_points, in_labels)

        if self.dict_format:
            inout = 1
            point_label = 1
            boxes = []
            box_old = []
            pt = np.array([0, 0])
            bboxes = []

            pt = points_for_image
            point_label = np.array(in_labels)

            name = image_path.split('/')[-1].split(".tif")[0]
            image_meta_dict = {'filename_or_obj': name}
            return {
                'image': input_image,
                'label': input_label,
                'p_label': point_label,
                'pt': pt,
                'box': boxes,
                # 'box_old':box_old,
                'image_meta_dict': image_meta_dict,
                'ground_truth_bboxes': bboxes
            }

        return input_image, input_label, points

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
        
    def resize_array(self,array, new_size):
        # Transpose the array to height x width x channels
        array = np.transpose(array, (1, 2, 0))

        array = array * 255
        # Convert to PIL Image
        pil_image = Image.fromarray(array.astype(np.uint8))

        # Resize the image
        resized_image = pil_image.resize(new_size)

        # Convert back to numpy array
        resized_array = np.array(resized_image)

        # Transpose back to channels x height x width
        resized_array = np.transpose(resized_array, (2, 0, 1))
        
        resized_array = resized_array/255

        return resized_array

class TestDataset():
    def __init__(self, image_id, label_id, is_robustness,dict_format=True):

        self.image_id = image_id
        self.label_id = label_id
        self.is_robustness = is_robustness
        self.dict_format=dict_format

        self.image_list = [f"./ai4boundaries_data/orthophoto/images/val/{x}_ortho_1m_512.tif" for x in self.image_id]
        self.label_list = [f"./ai4boundaries_data/orthophoto/masks/val/{x}_ortholabel_1m_512.tif" for x in self.image_id]


    def __getitem__(self, item):

        image_path = self.image_list[item]
        #image_path = os.path.join(self.image_path, image_name)
        input_image = self.__open_tiff__(image_path)
        input_image = self.min_max_normalize(input_image)
        input_image = self.resize_array(input_image,(1024,1024))
        #input_image = image.resize((1024, 1024), Image.ANTIALIAS)
        input_image = torch.tensor(input_image).float()

        label_path = self.label_list[item]
        #label_path = os.path.join(self.label_path, label_name)
        input_label = self.__open_tiff__(label_path)
        input_label = self.resize_array(input_label,(256,256))
        
        #label = label.resize((256, 256), Image.ANTIALIAS)

        input_label = torch.tensor(input_label).long()

        points_scale = np.array(input_image.shape[1:])[None, ::-1]
        point_grids = build_all_layer_point_grids(
            n_per_side=32,
            n_layers=0,
            scale_per_layer=1,
        )
        points_for_image = point_grids[0] * points_scale
        in_points = torch.as_tensor(points_for_image)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int)
        points = (in_points, in_labels)

        if self.dict_format:
            inout = 1
            point_label = 1
            boxes = []
            box_old = []
            pt = np.array([0, 0])
            bboxes = []

            pt = points_for_image
            point_label = np.array(in_labels)

            name = image_path.split('/')[-1].split(".tif")[0]
            image_meta_dict = {'filename_or_obj': name}
            return {
                'image': input_image,
                'label': input_label,
                'p_label': point_label,
                'pt': pt,
                'box': boxes,
                # 'box_old':box_old,
                'image_meta_dict': image_meta_dict,
                'ground_truth_bboxes': bboxes
            }

        return input_image, input_label, points

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
        
    def resize_array(self,array, new_size):
        # Transpose the array to height x width x channels
        array = np.transpose(array, (1, 2, 0))

        array = array * 255
        # Convert to PIL Image
        pil_image = Image.fromarray(array.astype(np.uint8))

        # Resize the image
        resized_image = pil_image.resize(new_size)

        # Convert back to numpy array
        resized_array = np.array(resized_image)

        # Transpose back to channels x height x width
        resized_array = np.transpose(resized_array, (2, 0, 1))
        
        resized_array = resized_array/255

        return resized_array

class Ai4smallDataset():
    def __init__(self,image_list,dict_format=True):

        self.image_list = image_list
        self.image_ids = [x.split("/")[-1].split(".")[0] for x in self.image_list]
        self.mask_list= [f"./original/sentinel-2-asia/parcel_mask/{x}.tif" for x in self.image_ids]
        self.dict_format=dict_format

    def __getitem__(self, item):

        image_path = self.image_list[item]
        #image_path = os.path.join(self.image_path, image_name)
        input_image = self.__open_tiff__(image_path)
        input_image = self.min_max_normalize(input_image)
        input_image = self.resize_array(input_image,(1024,1024))
        #input_image = image.resize((1024, 1024), Image.ANTIALIAS)
        input_image = torch.tensor(input_image).float()

        label_path = self.mask_list[item]
        #label_path = os.path.join(self.label_path, label_name)
        input_label = self.__open_tiff__(label_path)
        input_label = self.resize_array(input_label,(256,256),mask=True)
        input_label = torch.tensor(input_label)

        points_scale = np.array(input_image.shape[1:])[None, ::-1]
        point_grids = build_all_layer_point_grids(
            n_per_side=32,
            n_layers=0,
            scale_per_layer=1,
        )
        points_for_image = point_grids[0] * points_scale
        in_points = torch.as_tensor(points_for_image)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int)
        points = (in_points, in_labels)

        if self.dict_format:
            inout = 1
            point_label = 1
            boxes = []
            box_old = []
            pt = np.array([0, 0])
            bboxes = []

            pt = points_for_image
            point_label = np.array(in_labels)

            name = self.image_ids[item]
            image_meta_dict = {'filename_or_obj': name}
            return {
                'image': input_image[:3,:,:],
                'label': input_label,
                'p_label': point_label,
                'pt': pt,
                'box': boxes,
                # 'box_old':box_old,
                'image_meta_dict': image_meta_dict,
                'ground_truth_bboxes': bboxes
            }

        return input_image, input_label, points

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


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def get_images_and_labels_path(data_path, mode):

    label_list = sorted([os.path.join(data_path, mode, "labels", label_file) for label_file in os.listdir(os.path.join(data_path, mode, "labels"))])
    image_list = sorted([os.path.join(data_path, mode, "images", image_file) for image_file in os.listdir(os.path.join(data_path, mode, "images"))])

    print(mode, "data length:", len(label_list), len(image_list))

    return label_list, image_list
