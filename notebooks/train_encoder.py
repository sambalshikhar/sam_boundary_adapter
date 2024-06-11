import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from tensorboardX import SummaryWriter
from dataset import *
from SAM_conf import settings
from notebooks.SAM_conf import SAM_cfg
from torch.utils.data import DataLoader
from notebooks.SAM_conf.SAM_utils import *
import function
from glob import glob
args = SAM_cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

# Initialize wandb at the start of your script
wandb.login()
wandb.init(project="Custom decoder",name=args.exp_name,config=args)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights, strict=False)

if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']

    net.load_state_dict(checkpoint['state_dict'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('../logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

if args.dataset == 'ai4boundaries':
    df=pd.read_csv("./ai4boundaries_data/ai4boundaries_ftp_urls_all.csv")
    df_region=df[df['file_id'].str.contains("AT")]
    df_region_train=df_region[df_region['split']=='train']
    df_region_val=df_region[df_region['split']=='val']
    
    train_image = df_region_train['file_id'].tolist()
    train_label = df_region_train['file_id'].tolist()
    train_data = TrainDataset(train_image, train_label,is_robustness=False)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    val_image = df_region_val['file_id'].tolist()
    val_label = df_region_val['file_id'].tolist()
    val_data = TestDataset(val_image, val_label,is_robustness=False)
    val_dataloader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)
    '''end'''
if args.dataset=='ai4small':

    train_image_list=glob("../original/sentinel-2-asia/train/images/*")
    val_image_list=glob("../original/sentinel-2-asia/validate/images/*")

    train_data=Ai4smallDataset(train_image_list)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    val_data=Ai4smallDataset(val_image_list)
    val_dataloader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

    
'''checkpoint path and tensorboard'''

checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
    settings.LOG_DIR, args.net, settings.TIME_NOW))

# create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begin training'''
best_acc = 0.0
best_tol = 1e6
best_iou = 1e10
best_dice = 1e10

#optimizer = torch.optim.AdamW(net.parameters(),lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999),weight_decay=0.0)
total_iterations_per_epoch = len(train_dataloader)  # Adjust based on your requirements
T_max = total_iterations_per_epoch*10
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.0001,step_size_up=T_max,cycle_momentum=False)
scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=3e-5)

for epoch in range(300):
    if args.mod == 'sam_adpt':
        net.train()
        time_start = time.time()
        loss= function.train_sam(args, net, optimizer, train_dataloader, epoch,writer,scheduler,vis=args.vis)
        #current_lr=scheduler.get_last_lr()[0]
        logger.info(f'Train loss: {loss}|| lr :{optimizer.param_groups[0]["lr"]}  @ epoch {epoch}.')

        time_end = time.time()
        
        print('time_for_training ', time_end - time_start)
        net.eval()
        
        if True:
            tol, (eiou, edice) = function.validation_sam(args, val_dataloader, epoch, net,writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            wandb.log({"Epoch": epoch,"Train Loss": loss,"Learning Rate":optimizer.param_groups[0]["lr"],"Dice":edice})

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if tol < best_tol or eiou > best_iou or edice > best_dice:
                best_iou = eiou
                best_dice = edice
                best_tol = tol
                is_best = True

                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.net,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                    'best_tol': best_tol,
                    'path_helper': args.path_helper,
                }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
                print("-------------save best checkpoint------------------")
            else:
                is_best = False

writer.close()
