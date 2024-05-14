import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet_model import UNet_model

from monai.losses import DiceCELoss
from dataset import *
from torch.autograd import Function
import time
import logging
import argparse
from glob import glob
import segmentation_models_pytorch as smp
from tqdm import tqdm


def cal_iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

def eval_seg(pred, true_mask_p, threshold):

    eiou, edice = 0, 0
    for th in threshold:
        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()
        disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')

        disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')

        '''iou for numpy'''
        eiou += cal_iou(disc_pred, disc_mask)

        '''dice for torch'''
        edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

    return eiou / len(threshold), edice / len(threshold)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:{}'.format(total_num))
    print('trainable:{}'.format(trainable_num))
    # return {'Total': total_num, 'Trainable': trainable_num}


def evaluate(model,val_dataloader):
    model.eval()

    val_loss = []
    iou_list = []
    dice_list = []
    with torch.no_grad():

        for pack in tqdm(val_dataloader):
            
            image = pack['image'].to(dtype=torch.float32, device=device)

            masks = pack['label'].to(dtype=torch.float32, device=device)
            masks = masks[:,0,:,:]
            masks = masks.unsqueeze(0)

            image = image.to(device=device)
            label = masks.to(device=device)

            pred = model(image)

            loss = lossfunc(pred,label)
            val_loss.append(loss.item())
            iou,dice = eval_seg(pred, label, threshold)
            iou_list.append(iou)
            dice_list.append(dice)

        loss_mean = np.average(val_loss)
        iou_mean = np.average(iou_list)
        dice_mean = np.average(dice_list)

    return  loss_mean,iou_mean,dice_mean

def train(model,train_dataloader):

    model.train()
    train_loss = []
    iou_list = []
    dice_list = []
    for pack in tqdm(train_dataloader):
        image = pack['image'].to(dtype=torch.float32, device=device)

        masks = pack['label'].to(dtype=torch.float32, device=device)
        masks = masks[:,0,:,:]
        masks = masks.unsqueeze(0)
        
        image = image.to(device=device)
        label = masks.to(device=device)
        
        optimizer.zero_grad()

        pred = model(image)

        loss = lossfunc(pred, label) * 100
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        iou, dice = eval_seg(pred,label, threshold)
        iou_list.append(iou)
        dice_list.append(dice)

    loss_mean = np.average(train_loss)
    iou_mean = np.average(iou_list)
    dice_mean = np.average(dice_list)

    print(
        f"| epoch {epoch:3d} | "f"train loss {loss_mean:5.2f} | "f"iou {iou_mean:3.2f}  | "f"dice {dice_mean:3.2f}"
    )


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    model = model.to(device)
    
    train_image_list=glob("./original/sentinel-2-asia/train/images/*")
    val_image_list=glob("./original/sentinel-2-asia/validate/images/*")

    train_data=Ai4smallDataset(train_image_list)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    val_data=Ai4smallDataset(val_image_list)
    val_dataloader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = 100
    best_iou = 1
    best_dice = 1
    logger = logging.getLogger('unet')
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs=100
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train(model,train_dataloader)
        val_loss,iou,dice = evaluate(model,val_dataloader)

        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.4f} | "f"iou {iou:3.2f}  | "f"dice {dice:3.2f}" )
        print("-" * 89)

        if (val_loss < best_loss) or (iou > best_iou) or (dice > best_dice) :
            best_loss = val_loss
            best_iou = iou
            best_dice = dice
            torch.save(model.state_dict(), 'unet.pt')
            logger.info(f"Best model saved!")