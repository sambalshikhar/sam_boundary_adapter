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
            print("shape of output",outputs[1].size())
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
    model = SEANet(num_classes=1)

    model = model.to(device)
    
    model.apply(weights_init)
    load_vgg16pretrain(model)  #
    
    net_parameters_id = defaultdict(list)
    net = model
    for pname, p in net.named_parameters():
        print(pname)
        if pname in ['conv1_1.weight','conv1_2.weight',
                     'conv2_1.weight','conv2_2.weight',
                     'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                     'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
            # print(pname, 'lr:1 de:1')
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)

        elif pname in ['conv_final1.weight','conv_final2.weight']:
            # print(pname, 'lr:2 de:0')
            net_parameters_id['final1-2.weight'].append(p)
        elif pname in ['conv_final1.bias','conv_final2.bias']:
            # print(pname, 'lr:2 de:0')
            net_parameters_id['final1-2.bias'].append(p)

        elif pname in ['conv1_1.bias','conv1_2.bias',
                       'conv2_1.bias','conv2_2.bias',
                       'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                       'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
            # print(pname, 'lr:2 de:0')
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
            # print(pname, 'lr:100 de:1')
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :
            # print(pname, 'lr:200 de:0')
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                       'conv2_1_down.weight','conv2_2_down.weight',
                       'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                       'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                       'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
            # print(pname, 'lr:0.1 de:1')
            if 'conv_down_1-5.weight' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.weight'] = []
            net_parameters_id['conv_down_1-5.weight'].append(p)
        elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                       'conv2_1_down.bias','conv2_2_down.bias',
                       'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                       'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                       'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
            # print(pname, 'lr:0.2 de:0')
            if 'conv_down_1-5.bias' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.bias'] = []
            net_parameters_id['conv_down_1-5.bias'].append(p)
        elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                       'score_dsn4.weight','score_dsn5.weight']:
            # print(pname, 'lr:0.01 de:1')
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                       'score_dsn4.bias','score_dsn5.bias']:
            # print(pname, 'lr:0.02 de:0')
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            # print(pname, 'lr:0.001 de:1')
            if 'score_final.weight' not in net_parameters_id:
                net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'score_final.bias' not in net_parameters_id:
                net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)

        elif pname in ['aspp.convs.0.0.weight','aspp.convs.0.1.weight','aspp.convs.1.0.weight','aspp.convs.1.1.weight','aspp.convs.2.0.weight',
                       'aspp.convs.2.1.weight','aspp.convs.3.0.weight','aspp.convs.3.1.weight','aspp.convs.4.1.weight','aspp.convs.4.2.weight',
                       'aspp.project.0.weight','aspp.project.1.weight']:
            # print(pname, 'lr:0.002 de:0')
            if 'aspp1-12.weight' not in net_parameters_id:
                net_parameters_id['aspp1-12.weight'] = []
            net_parameters_id['aspp1-12.weight'].append(p)
        elif pname in ['aspp.convs.0.1.bias','aspp.convs.1.1.bias','aspp.convs.2.1.bias','aspp.convs.3.1.bias',
                       'aspp.convs.4.2.bias','aspp.project.1.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'aspp1-6.bias' not in net_parameters_id:
                net_parameters_id['aspp1-6.bias'] = []
            net_parameters_id['aspp1-6.bias'].append(p)

        elif pname in ['aspp1.convs.0.0.weight','aspp1.convs.0.1.weight','aspp1.convs.1.0.weight','aspp1.convs.1.1.weight','aspp1.convs.2.0.weight',
                       'aspp1.convs.2.1.weight','aspp1.convs.3.0.weight','aspp1.convs.3.1.weight','aspp1.convs.4.1.weight','aspp1.convs.4.2.weight',
                       'aspp1.project.0.weight','aspp1.project.1.weight']:
            # print(pname, 'lr:0.002 de:0')
            if 'as1-12.weight' not in net_parameters_id:
                net_parameters_id['as1-12.weight'] = []
            net_parameters_id['as1-12.weight'].append(p)
        elif pname in ['aspp1.convs.0.1.bias','aspp1.convs.1.1.bias','aspp1.convs.2.1.bias','aspp1.convs.3.1.bias',
                       'aspp1.convs.4.2.bias','aspp1.project.1.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'as1-6.bias' not in net_parameters_id:
                net_parameters_id['as1-6.bias'] = []
            net_parameters_id['as1-6.bias'].append(p)

        elif pname in ['center.block.1.conv.weight','dec5.block.1.conv.weight','dec5.block.2.conv.weight','dec4.block.1.conv.weight','dec4.block.2.conv.weight','dec3.block.2.conv.weight','dec3.block.1.conv.weight',
                       'dec2.block.2.conv.weight','dec2.block.1.conv.weight','dec1.conv.weight',' xdf.conv.weight','center.block.2.conv.weight',
                       'center.block.3.fc.0.weight','center.block.3.fc.2.weight','dec5.block.3.fc.0.weight','dec5.block.3.fc.2.weight','dec4.block.3.fc.0.weight',
                       'dec4.block.3.fc.2.weight','dec3.block.3.fc.0.weight','dec3.block.3.fc.2.weight','dec2.block.3.fc.0.weight','dec2.block.3.fc.2.weight']:
            # print(pname, 'lr:0.002 de:0')
            if 'dec1-22.weight' not in net_parameters_id:
                net_parameters_id['dec1-22.weight'] = []
            net_parameters_id['dec1-22.weight'].append(p)

        elif pname in ['center.block.1.conv.bias','dec5.block.1.conv.bias','dec5.block.2.conv.bias','dec4.block.1.conv.bias','dec4.block.2.conv.bias','dec3.block.2.conv.bias','dec3.block.1.conv.bias',
                       'dec2.block.2.conv.bias','dec2.block.1.conv.bias','dec1.conv.bias',' xdf.conv.bias','center.block.2.conv.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'dec1-12.bias' not in net_parameters_id:
                net_parameters_id['dec1-12.bias'] = []
            net_parameters_id['dec1-12.bias'].append(p)

    epoch_start = "0"
    
        
    train_image_list=glob("/home/geospatial/sambal/sam_boundary_adapter/original/sentinel-2-asia/train/images/*")
    val_image_list=glob("/home/geospatial/sambal/sam_boundary_adapter/original/sentinel-2-asia/validate/images/*")

    train_data=Ai4smallDataset(train_image_list,'train')
    trainLoader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)

    val_data=Ai4smallDataset(val_image_list,'validate')
    valLoader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

    optimizer = torch.optim.SGD([
        {'params': net_parameters_id['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
        {'params': net_parameters_id['conv5.weight']        , 'lr': args.lr*10  , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['conv5.bias']          , 'lr': args.lr*20  , 'weight_decay': 0.},
        {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': args.lr*0.1  , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': args.lr*0.2  , 'weight_decay': 0.},
        {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
        {'params': net_parameters_id['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
        {'params': net_parameters_id['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
    ], lr=args.lr, momentum=args.momentum,  weight_decay=args.weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  #  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)
   # scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.1)    #新加的
    criterion = define_loss(args.model_type)


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
        scheduler.step()

        epoch_loss = running_loss / len(train_image_list)
        #print(epoch_loss)

        iou,dice= evaluate(device, epoch, model, valLoader, writer)

        logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), os.path.join(args.save_path, str(epoch) + ".pt")
            )


