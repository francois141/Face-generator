import torch
import torch.nn as nn
import math

class Generator_block(nn.Module):
    def __init__(self,in_layers,out_layers):
        super(Generator_block, self).__init__()
        self.block = nn.Sequential()

        # This is a decoder with convtranspose - batchnorm - blocks
        self.block.add_module("Conv0", nn.ConvTranspose2d(in_layers, out_layers, 4, 2, 1, bias=False))
        self.block.add_module("Batchnorm0", nn.BatchNorm2d(out_layers))
        self.block.add_module("ReLU0", nn.ReLU(True))

        for i in range(1,3):
            self.block.add_module("Conv{0}".format(i),nn.Conv2d(out_layers, out_layers,kernel_size=3,padding=1))
            self.block.add_module("Batchnorm{0}".format(i),nn.BatchNorm2d(out_layers))
            self.block.add_module("ReLU{0}".format(i),nn.ReLU(True))

    def forward(self,x):
        return self.block(x)
    
class Final_block(nn.Module):

    def __init__(self,in_layers):
        super(Final_block, self).__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(in_layers, 3, 4, 2, 1, bias=False))

    def forward(self,x):
        return self.block(x)


# Creation of a simple generator
class Generator(nn.Module):
    def __init__(self,latent_space_size):
        # Initialize the module
        super(Generator, self).__init__()

        # We can parametrize the size of the latent space
        self.latent_stace_size = latent_space_size

        # Some variables for the gan
        self.nb_channels = 3
        self.final_feature_map_size = 64
        self.depth = int(math.log2(self.final_feature_map_size))
        
        # Building the network with the layers
        layers = []
        nb_convs = 2**(self.depth-2)

        # First layer
        layers.append(Generator_block(self.latent_stace_size,self.final_feature_map_size*nb_convs))

        # Intermediate layers
        for _ in range(self.depth-2):
            layers.append(Generator_block(self.final_feature_map_size*nb_convs,self.final_feature_map_size*nb_convs//2))
            nb_convs //= 2

        # Last layer
        layers.append(Final_block(self.final_feature_map_size))

        # Creation of the network
        self.network = nn.Sequential(*layers)

        
    def forward(self, input):
        return self.network(input)

    def get_weights(self):
        return self.state_dict()

    def set_weights(self,weights):
        return self.load_state_dict(weights)

    # We can save the weights for the demo
    def save_weights(self,path):
        return torch.save(self.state_dict(), path)

    # Load the weights for the demo
    def load_weights(self,path):
        self.load_state_dict(torch.load(path))