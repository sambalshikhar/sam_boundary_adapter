import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from dataset import *
from torch.utils.data import DataLoader
from SAM_conf.SAM_utils import *
import function

args = SAM_cfg.parse_args()
if args.dataset == 'refuge' or args.dataset == 'refuge2':
    args.data_path = '../dataset'

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

'''load pretrained model'''
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)
checkpoint = torch.load(checkpoint_file, map_location=loc)
start_epoch = checkpoint['epoch']
best_tol = checkpoint['best_tol']

state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = 'module.' + k
        new_state_dict[name] = v
    # load params
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict, strict=False)

args.path_helper = checkpoint['path_helper']
logger = create_logger(args.path_helper['log_path'])
print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('../logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

df=pd.read_csv("/home/geovisionaries/sambal/sam_boundary_adapter/ai4boundaries_data/ai4boundaries_ftp_urls_all.csv")
df_region=df[df['file_id'].str.contains("AT")]
df_region_test=df_region[df_region['split']=='test']

test_image = df_region_test['file_id'].tolist()
test_label = df_region_test['file_id'].tolist()
test_data = TestDataset(test_image, test_label,is_robustness=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)
'''end'''

'''begain valuation'''
best_acc = 0.0
best_tol = 1e4

if args.mod == 'sam_adpt':
    net.eval()
    tol, (eiou, edice) = function.Test_sam(args, test_dataloader, start_epoch, net)
    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')
