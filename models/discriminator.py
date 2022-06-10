import torch
import torch.nn as nn

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
            nn.Conv2d(self.nb_channels, self.final_feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Dimensions at this point : 64 x 32 x 32
            nn.Conv2d(self.final_feature_map_size, self.final_feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.final_feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.final_feature_map_size * 2, self.final_feature_map_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.final_feature_map_size * 2, self.final_feature_map_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Dimensions at this point : 128 x 16 x 16
            nn.Conv2d(self.final_feature_map_size * 2, self.final_feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.final_feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.final_feature_map_size * 4, self.final_feature_map_size * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.final_feature_map_size * 4, self.final_feature_map_size * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Dimensions at this point : 256 x 8 x 8
            nn.Conv2d(self.final_feature_map_size * 4, self.final_feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.final_feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.final_feature_map_size * 8, self.final_feature_map_size * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.final_feature_map_size * 8, self.final_feature_map_size * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.final_feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Dimensions at this point : 512 x 4 x 4
            nn.Conv2d(self.final_feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Dimensions at this point : 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)

    # We can save the weights for the demo
    def save_weights(self,path):
        return torch.save(self.state_dict(), path)

    # Load the weights for the demo
    def load_weights(self,path):
        self.load_state_dict(torch.load(path))

