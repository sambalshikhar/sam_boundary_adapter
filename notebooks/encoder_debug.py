import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from dataset import *
from SAM_conf import settings
from notebooks.SAM_conf import SAM_cfg
from torch.utils.data import DataLoader
from notebooks.SAM_conf.SAM_utils import *
import function
from glob import glob
from models_samus.model_dict import get_model
args = SAM_cfg.parse_args()
print(args)
GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
input_image=torch.rand((1,3,1024,1024)).to(dtype=torch.float32, device=GPUdevice)
net.eval()
with torch.no_grad():
    x=net.image_encoder(input_image)
    print(x.size())
    