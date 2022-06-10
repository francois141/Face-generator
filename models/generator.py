import torch
import torch.nn as nn

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

        # Creation of the network
        self.network = nn.Sequential(
            # This is a decoder with Convtranspose - batchnorm - blocks
            nn.ConvTranspose2d( self.latent_stace_size, self.final_feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.final_feature_map_size * 8),
            nn.ReLU(True),

            nn.Conv2d(self.final_feature_map_size * 8, self.final_feature_map_size*8,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 8),
            nn.ReLU(True),
            nn.Conv2d(self.final_feature_map_size * 8, self.final_feature_map_size*8,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 8),
            nn.ReLU(True),

            # Dimensions at this point : 512 x 4 x 4
            nn.ConvTranspose2d(self.final_feature_map_size * 8, self.final_feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.final_feature_map_size * 4),
            nn.ReLU(True),

            nn.Conv2d(self.final_feature_map_size * 4, self.final_feature_map_size*4,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 4),
            nn.ReLU(True),
            nn.Conv2d(self.final_feature_map_size * 4, self.final_feature_map_size*4,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 4),
            nn.ReLU(True),

            # Dimensions at this point : 256 x 8 x 8
            nn.ConvTranspose2d( self.final_feature_map_size * 4, self.final_feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.final_feature_map_size * 2),
            nn.ReLU(True),

            nn.Conv2d(self.final_feature_map_size * 2, self.final_feature_map_size*2,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 2),
            nn.ReLU(True),
            nn.Conv2d(self.final_feature_map_size * 2, self.final_feature_map_size*2,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 2),
            nn.ReLU(True),

            # Dimensions at this point : 128 x 16 x 16
            nn.ConvTranspose2d( self.final_feature_map_size * 2, self.final_feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.final_feature_map_size),
            nn.ReLU(True),

            nn.Conv2d(self.final_feature_map_size , self.final_feature_map_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.final_feature_map_size),
            nn.ReLU(True),
            nn.Conv2d(self.final_feature_map_size, self.final_feature_map_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.final_feature_map_size),
            nn.ReLU(True),


            # Dimensions at this point : 64 x 32 x 32
            nn.ConvTranspose2d( self.final_feature_map_size, self.nb_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Dimensions at this point : 3 x 64 x 64
        )

        
    def forward(self, input):
        # Perform a single feed forward inside of the network
        return self.network(input)
        

    # We can save the weights for the demo
    def save_weights(self,path):
        return torch.save(self.state_dict(), path)

    # Load the weights for the demo
    def load_weights(self,path):
        self.load_state_dict(torch.load(path))


