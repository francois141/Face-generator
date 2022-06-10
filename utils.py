import torch
import torch.nn as nn
import random

# This function will return a GPU device, if available
def get_device():
    return torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# This function can initialize the weights of the discriminator and the generator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)