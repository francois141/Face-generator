import torch
import torch.nn as nn
import math

class First_block(nn.Module):
    def __init__(self,nb_channels,out_layers):
        super(First_block, self).__init__()

        self.conv = nn.Conv2d(nb_channels, out_layers, 4, 2, 1, bias=False)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.block = nn.Sequential()
        self.block.add_module("Conv_0",self.conv)
        self.block.add_module("Activation_0",self.activation)

    def forward(self,x):
        return self.block(x)

class Discriminator_block(nn.Module):
    def __init__(self,in_layers,out_layers):
        super(Discriminator_block, self).__init__()
        self.block = nn.Sequential()

        # This is an encoder with conv - batchnorm - blocks
        self.block.add_module("Conv_0",nn.Conv2d(in_layers,out_layers, 4, 2, 1, bias=False))
        self.block.add_module("Batchnorm_0",nn.BatchNorm2d(out_layers))
        self.block.add_module("ReLU_0",nn.LeakyReLU(0.2, inplace=True))

        for i in range(1,3):
            self.block.add_module("Conv_{0}".format(i),nn.Conv2d(out_layers, out_layers,kernel_size=3,padding=1))
            self.block.add_module("Batchnorm_{0}".format(i),nn.BatchNorm2d(out_layers))
            self.block.add_module("ReLU_{0}".format(i),nn.ReLU(True))

    def forward(self,x):
        return self.block(x)


class Last_block(nn.Module):
    def __init__(self,in_layers):
        super(Last_block, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_layers, 1, 4, 1, 0, bias=False),nn.Sigmoid())

    def forward(self,x):

        x = self.block(x)
        # [batch_size,1,1,1] -> [batch_size]
        return x.reshape(-1)


# Creation of a simple discriminator
class Discriminator(nn.Module):
    def __init__(self):
        # Initialize the module
        super(Discriminator, self).__init__()

        # Some variables for the gan
        self.nb_channels = 3
        self.final_feature_map_size = 64
        self.depth = int(math.log2(self.final_feature_map_size))

        self.first_block = First_block(self.nb_channels,self.final_feature_map_size)
        self.last_block = Last_block(self.final_feature_map_size*8)

        # Building the network with the layers
        layers = []

        # First layer
        layers.append(self.first_block)

        # Intermediate layers
        for i in range(self.depth - 3):
            in_layers,out_layers  = 2**i, 2**(i+1)
            layers.append(Discriminator_block(self.final_feature_map_size*in_layers,self.final_feature_map_size*out_layers))

        # Last layer
        layers.append(self.last_block)

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
