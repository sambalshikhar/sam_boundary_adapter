from seanet import SEANet
import torch
x=torch.rand((4,3,512,512)).float().cuda()
model = SEANet(num_classes=1).cuda()
print(model(x)[0].size())