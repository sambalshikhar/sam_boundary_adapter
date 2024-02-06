import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet_model import UNet_model
# from ori_sam_model import Sam_model
from segment_anything import sam_model_registry
from monai.losses import DiceCELoss
from dataset import TrainDataset,TestDataset
from torch.autograd import Function
import time
import logging
import argparse
from tqdm import tqdm
import pandas as pd
import wandb
class Sam_model(nn.Module):
    def __init__(self,model_type,sam_checkpoint):
        super(Sam_model, self).__init__()

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    def forward(self, x, points):

        image = self.sam.image_encoder(x)
        se, de = self.sam.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
        pred, _ = self.sam.mask_decoder(
            image_embeddings=image,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=False,
        )

        return pred



class Model(nn.Module):
    def __init__(self,model_type,sam_checkpoint):
        super(Model,self).__init__()

        self.unet = UNet_model(in_channels=3,out_channels=3)
        self.sam = Sam_model(model_type=model_type,sam_checkpoint=sam_checkpoint)

        for n, value in self.sam.named_parameters():
            if 'mask_decoder' in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def forward(self, x, points):

        denoised_img = self.unet(x)
        img_add = x + denoised_img
        img_add = torch.clamp(img_add,0,255)
        masks = self.sam(img_add,points)
        return denoised_img, masks

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

        for image,label,points in tqdm(val_dataloader):

            image = image.to(device=device)
            label = label.to(device=device)
            label = label[:,0,:,:].unsqueeze(0)
            denoised_img, pred = model(image, points)

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
    for image, label, points in tqdm(train_dataloader):
        image = image.to(device=device)
        label = label.to(device=device)
        label = label[:,0,:,:].unsqueeze(0)
        optimizer.zero_grad()

        denoised_img, pred = model(image, points)

        loss = lossfunc(pred, label) * 100
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        iou, dice = eval_seg(pred, label, threshold)
        iou_list.append(iou)
        dice_list.append(dice)

    loss_mean = np.average(train_loss)
    iou_mean = np.average(iou_list)
    dice_mean = np.average(dice_list)

    print()
    print(f"| epoch {epoch:3d} | "f"train loss {loss_mean:5.2f} | "f"iou {iou_mean:3.2f}  | "f"dice {dice_mean:3.2f}")
    
    return loss_mean,iou_mean,dice_mean
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', type=str, required=True, help='the name of your dataset')
    parser.add_argument('-exp_name', type=str, required=True, help='the name of your experiment')
    parser.add_argument('-bs', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-epochs', type=int, default=100, help='the number of training sessions')
    parser.add_argument('-lr', type=int, default=0.00005, help='learning rate')
    parser.add_argument('-model_type', type=str, default="vit_h", help='')
    parser.add_argument('-sam_ckpt', default='/home/prateekjha/sam_boundary_adapter/checkpoint/sam_vit_h_4b8939.pth', type=str, help='sam checkpoint path')
    parser.add_argument('-save_path', type=str, required=True, help='the path to save your training result')
    args = parser.parse_args()


    # start a new experiment
    wandb.init(project="sam_adapter_head")

    df=pd.read_csv("/home/prateekjha/ai4boundaries_data/sampling/ai4boundaries_ftp_urls_all.csv")
    df_region=df[df['file_id'].str.contains("AT")]
    df_region_train=df_region[df_region['split']=='train']
    df_region_val=df_region[df_region['split']=='val']
    
    train_image = df_region_train['file_id'].tolist()
    train_label = df_region_train['file_id'].tolist()
    train_data = TrainDataset(train_image, train_label,is_robustness=False,dict_format=True)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    val_image = df_region_val['file_id'].tolist()
    val_label = df_region_val['file_id'].tolist()
    val_data = TestDataset(val_image, val_label,is_robustness=False,dict_format=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = 100
    best_iou = 1
    best_dice = 1
    logger = logging.getLogger('head-prompt SAM')

    model = Model(model_type=args.model_type, sam_checkpoint=args.sam_ckpt)
    model = model.to(device)
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss_mean,train_iou_mean,train_dice_mean=train(model,train_dataloader)
        val_loss,iou,dice = evaluate(model,val_dataloader)

        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.4f} | "f"iou {iou:3.2f}  | "f"dice {dice:3.2f}" )
        print("-" * 89)
        
        wandb.log({"test_loss": train_loss_mean,
                   "train_iou":train_iou_mean,
                    "train_dice":train_dice_mean,
                    "val_loss":val_loss,
                    "val_iou":iou,
                    "val_dice":dice})

        if (val_loss < best_loss) or (iou > best_iou) or (dice > best_dice) :
            best_loss = val_loss
            best_iou = iou
            best_dice = dice
            torch.save(model.state_dict(), f'{args.save_path}/head_prompt_{args.data_name}_{args.exp_name}.pt')
            print(f"Best model saved!")
    
            

    print(f'Head prompt {args.data_name} {args.exp_name} training is complete ! ')