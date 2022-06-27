import torch
import torch.nn as nn


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
        # We can simply run the block
        return self.block(x)
    
class Final_block(nn.Module):
    def __init__(self,in_layers):
        super(Final_block, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_layers, 1, 4, 1, 0, bias=False),nn.Sigmoid())

    def forward(self,x):
        # We can simply run  the block
        return self.block(x)


# Creation of a simple discriminator
class Discriminator(nn.Module):
    def __init__(self):
        # Initialize the module
        super(Discriminator, self).__init__()

        # Some variables for the gan
        self.nb_channels = 3
        self.final_feature_map_size = 64

        # Creation of the network
        self.main = nn.Sequential(
            # This is a discriminator with Convtranspose - batchnorm - blocks
            Discriminator_block(self.nb_channels,self.final_feature_map_size),
            Discriminator_block(self.final_feature_map_size,self.final_feature_map_size*2),
            Discriminator_block(self.final_feature_map_size*2,self.final_feature_map_size*4),
            Discriminator_block(self.final_feature_map_size*4,self.final_feature_map_size*8),
            Final_block(self.final_feature_map_size*8),
        )

    def forward(self, input):
        return self.main(input)

    # We can save the weights for the demo
    def save_weights(self,path):
        return torch.save(self.state_dict(), path)

    # Load the weights for the demo
    def load_weights(self,path):
        self.load_state_dict(torch.load(path))

