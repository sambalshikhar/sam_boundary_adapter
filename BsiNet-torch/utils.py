import torch
from tqdm import tqdm
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse
from torch.autograd import Function

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
    #pred=torch.nn.functional.sigmoid(pred)
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

def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    iou_list=[]
    dice_list=[]
    threshold = [0.5]
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            inputs,mask,parcel= data
            inputs = inputs.to(device)
            parcel = parcel.to(device)
            pred_mask,pred_parcel = model(inputs)
            iou,dice=eval_seg(pred_parcel,parcel,threshold)
            iou_list.append(iou)
            dice_list.append(dice)
        iou_mean = np.average(iou_list)
        dice_mean = np.average(dice_list)

        print(f"Dev_metrics IoU:{iou_mean} dice:{dice_mean} at epoch {epoch}")

    return iou_mean,dice_mean


def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=True):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _,_ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)

            output_mask = outputs[0].detach().cpu().numpy()
            output_final = np.argmax(output_mask, axis=1).astype(float)
            output_final = torch.from_numpy(output_final).unsqueeze(1)

            if train == "True":
                save_image(targets.float(), "Target_train",val_batch_size)
                save_image(output_final, "Prediction_train",val_batch_size)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

            break


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")

    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=4, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument("--save_path", type=str, help="Model save path.")
        
    parser.add_argument('--lr', '--learning_rate', default=1e-8, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr1', '--learning_rate1', default=1e-2, type=float,
                        metavar='LR1', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                        metavar='W', help='default weight decay')
    parser.add_argument('--stepsize', default=15, type=int,
                        metavar='SS', help='learning rate step size')
    parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                        help='learning rate decay parameter: Gamma')

    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--test_path", type=str, help="path to img tif files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")

    return parser


