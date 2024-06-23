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
import wandb

#wandb.login()
#wandb.init(project="Custom decoder",name="deeplab_efficientnetb7")

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

def compute_metrics(preds, labels, threshold=0.5):
    """
    Compute the precision, recall, and F1 score for predicted masks.
    
    Args:
    preds (torch.Tensor): Predicted masks with shape (batch_size, 1, H, W).
    labels (torch.Tensor): Ground truth masks with shape (batch_size, 1, H, W).
    threshold (float): Threshold to convert probabilities to binary masks.
    
    Returns:
    dict: Dictionary containing average precision, recall, and F1 score.
    """
    # Threshold the predictions
    preds = preds > threshold
    
    # Flatten the tensors to simplify calculation
    preds = preds.view(preds.shape[0], -1).float()
    labels = labels.view(labels.shape[0], -1).float()
    
    # True positives, false positives, and false negatives
    TP = (preds * labels).sum(dim=1)
    FP = (preds * (1 - labels)).sum(dim=1)
    FN = ((1 - preds) * labels).sum(dim=1)
    
    # Precision, recall, and F1 for each element in the batch
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Average across the batch
    avg_precision = precision.mean().item()
    avg_recall = recall.mean().item()
    avg_f1 = f1.mean().item()

    return avg_precision,avg_recall,avg_f1

def evaluate(model,val_dataloader):
    model.eval()

    val_loss = []
    iou_list = []
    dice_list = []

    pr_list=[]
    re_list=[]
    f1_list=[]
    with torch.no_grad():

        for pack in tqdm(val_dataloader):
            
            image = pack['image'].to(device=device)

            masks = pack['label'].to(device=device)
            #masks = masks.unsqueeze(1)



            image = image.to(device=device)
            label = masks.to(device=device)

            pred = model(image)

            loss = lossfunc(pred,label)
            val_loss.append(loss.item())
            iou,dice = eval_seg(pred, label, threshold)
            pr,re,f1=compute_metrics(pred,label)

            pr_list.append(pr)
            re_list.append(re)
            f1_list.append(f1)

            iou_list.append(iou)
            dice_list.append(dice)

        loss_mean = np.average(val_loss)
        iou_mean = np.average(iou_list)
        dice_mean = np.average(dice_list)

        pr_mean = np.average(pr_list)
        re_mean = np.average(re_list)
        f1_mean = np.average(f1_list)




    return  loss_mean,iou_mean,dice_mean,pr_mean,re_mean,f1_mean

def train(model,train_dataloader):

    model.train()
    train_loss = []
    iou_list = []
    dice_list = []
    for pack in tqdm(train_dataloader):

        image = pack['image'].to(device=device)
        masks = pack['label'].to(device=device)
        #masks = masks.unsqueeze(1)

        image = image.to(device=device)
        label = masks.to(device=device)
        
        optimizer.zero_grad()

        pred = model(image)
        print(pred)

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

    return loss_mean


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = smp.Unet(
        encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,
        #activation='sigmoid',          # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    model = model.to(device)
    
    train_image_list=glob("../original/sentinel-2-asia/train/images/*")
    val_image_list=glob("../original/sentinel-2-asia/validate/images/*")

    train_data=Ai4smallDataset(train_image_list)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    val_data=Ai4smallDataset(val_image_list,split='validate')
    val_dataloader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)


    """
    df=pd.read_csv("./ai4boundaries_data/ai4boundaries_ftp_urls_all.csv")
    df_region=df[df['file_id'].str.contains("AT")]
    df_region_train=df_region[df_region['split']=='train']
    df_region_val=df_region[df_region['split']=='val']
    
    train_image = df_region_train['file_id'].tolist()
    train_label = df_region_train['file_id'].tolist()
    train_data = TrainDataset(train_image, train_label,is_robustness=False)
    train_dataloader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)

    val_image = df_region_val['file_id'].tolist()
    val_label = df_region_val['file_id'].tolist()
    val_data = TestDataset(val_image, val_label,is_robustness=False)
    val_dataloader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = 100
    best_iou = 1
    best_dice = 1
    logger = logging.getLogger('unet')
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs=200
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss=train(model,train_dataloader)
        val_loss,iou,dice,pr_mean,re_mean,f1_mean = evaluate(model,val_dataloader)

        #wandb.log({"Epoch": epoch,"Val Loss":val_loss,"Train loss":train_loss,"Learning Rate":optimizer.param_groups[0]["lr"],"Dice":dice})

        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.4f} | "f"iou {iou:3.2f}  | "f"dice {dice:3.2f}" 
            f"| f1 {f1_mean:5.2f} | precision {pr_mean:3.2f}  | recall {re_mean:3.2f}")
        print("-" * 89)

        if (val_loss < best_loss) or (iou > best_iou) or (dice > best_dice) :
            best_loss = val_loss
            best_iou = iou
            best_dice = dice
            torch.save(model.state_dict(), 'unet.pt')
            logger.info(f"Best model saved!")