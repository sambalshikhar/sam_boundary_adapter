from glob import glob
import logging
import os
import random
import torch
from dataset import Ai4smallDataset
from losses import LossBsiNet
from models import BsiNet
from tensorboardX import SummaryWriter
from torch import nn

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import visualize, create_train_arg_parser,evaluate
# from torchsummary import summary
from sklearn.model_selection import train_test_split
from torch.autograd import Function
import numpy as np
from faunet import FAUNet
from seanet import SEANet
import scipy.io as sio
from collections import defaultdict
import wandb

wandb.login()
wandb.init(project="Custom decoder",name="BSInet")

def define_loss(loss_type, weights=[1, 1, 1]):

    if loss_type == "bsinet":
        criterion = LossBsiNet(weights)

    return criterion


def build_model(model_type):

    if model_type == "bsinet":
        model = BsiNet(num_classes=1)

    return model


def train_model(model, targets, model_type, criterion, optimizer):

    if model_type == "bsinet":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(
                outputs[0], outputs[1],targets[0], targets[1]
            )
            loss.backward()
            optimizer.step()

    return loss

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()


#RCF预训练文件
def load_vgg16pretrain(model, vggmodel='./vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params = model.state_dict()

    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)




if __name__ == "__main__":

    args = create_train_arg_parser().parse_args()
    # args.val_path = './XJ_goole/test/image/'
    args.model_type = 'bsinet'
    args.save_path = './model'

    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    log_path = args.save_path + "/summary"
    writer = SummaryWriter(log_dir=log_path)
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    
    logging.basicConfig(
        filename="".format(args.object_type),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info("")


    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type)

    model = model.to(device)

    epoch_start = "0"    

    train_image_list=glob("/home/geovisionaries/sambal/original/sentinel-2-asia/train/images/*")
    val_image_list=glob("/home/geovisionaries/sambal/original/sentinel-2-asia/validate/images/*")

    train_data=Ai4smallDataset(train_image_list,flag='train')
    trainLoader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    val_data=Ai4smallDataset(val_image_list,flag='validate')
    valLoader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)
   # scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.1)    #新加的
    criterion = define_loss(args.model_type)

    args.num_epochs=1000

    for epoch in tqdm(
        range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs)
    ):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        for i, (inputs, targets1, targets2) in enumerate(
            tqdm(trainLoader)
        ):

            model.train()

            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets = [targets1, targets2]


            loss = train_model(model, targets, args.model_type, criterion, optimizer)

            #writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)
        #scheduler.step()

        epoch_loss = running_loss / len(train_image_list)
        #print(epoch_loss)

        iou,dice= evaluate(device, epoch, model, valLoader, writer)

        wandb.log({"Train Loss": epoch_loss,"Dice":dice})

        logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), os.path.join(args.save_path, str(epoch) + ".pt")
            )


