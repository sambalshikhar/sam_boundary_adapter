import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from torch.utils import data


from FAUNet_Train import channel_normalize
from FAUNet_Train import myDataset
from FAUNet_Train import DoubleConv,Down,Up,OutConv
from FAUNet_Train import FAUNet
from FAUNet_Train import mynormalize



def resize_image_cv2(image_array, target_size=(256, 256)):
    """
    Resize an image to a target size using OpenCV.
    
    Parameters:
    - image_array: numpy array of shape (height, width, channels).
    - target_size: tuple, target size in (width, height).
    
    Returns:
    - Resized image as a numpy array.
    """
    resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)
    return resized_image


def get_prediction_map(im,weight_path,Device='cuda:0'):
        # faunet = FAUNet(3,2,Device)
    faunet = torch.load(weight_path)
    # faunet.load_state_dict(PATH)
    faunet.eval()
    faunet.to(torch.device(Device))
    im = resize_image_cv2(im)
    
    im_transposed = np.transpose(im,(2,0,1))
    im_tensor = torch.tensor(im_transposed).float()     
    x =  im_tensor.to(torch.device(Device)).unsqueeze(0)
    c,e = faunet(x)
    seg = c.squeeze(0).cpu().numpy()
    edges = e.squeeze(0).cpu().numpy()       
    
    final_res =   1.0*(seg[1,:,:]>0)* (1 - 1.0*(edges[1,:,:]>0))
    final_res = final_res*255.0
    
    return final_res

