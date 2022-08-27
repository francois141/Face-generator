from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from models.discriminator import *
from models.generator import *
from utils import *
from datasets import *
import os

def main(args=None):

    # With parser we can change the configuration at run time
    parser = argparse.ArgumentParser(description='Training script for this GAN implementation')

    parser.add_argument('--dataroot',dest ='dataroot',type=str,default="celeb")
    parser.add_argument('--batch_size',dest ='batch_size',type=int,default=32)
    parser.add_argument('--num_epochs',dest ='num_epochs',type=int,default=10)
    parser.add_argument('--latent_space_size',dest='latent_space_size',type=int,default=100)
    parser.add_argument('--lr',dest ='lr',type=float,default=0.0002)
    parser.add_argument('--seed',dest ='seed',type=int,default=999)
    parser.add_argument('--save_weights',dest="save_weights",type=bool,default=True)
    parser.add_argument('--output_dir',dest="output_dir",type=str,default="output")
    parser.add_argument('--checkpoints_dir',dest="checkpoints_dir",type=str,default="checkpoints")
    parser.add_argument('--device',dest="device",type=str,default="")

    args = parser.parse_args(args)

    # Check if output directory exists
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # Check if checkpoint directory exists
    if not os.path.isdir(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)

    # We set the seed to be able to reproduce the experiments
    set_seed(args.seed)

    # Set the size of the image
    image_size = 64

    # Get the dataset
    dataset = get_images_dataset(args.dataroot, image_size)

    # Creation of the dataloader
    dataloader = get_dataloader(dataset,args.batch_size)

    # Decide which device we want to run on
    if(args.device == ""):
        device = get_device()
    else:
        device = args.device

    # Create a generator instance
    generator = Generator(args.latent_space_size).to(device)
    generator.apply(weights_init)

    # Create a discriminator instance
    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    # Setup Adam optimizers for both G and D 
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_generator = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # We use a classical cross-entropy loss in the gan
    criterion = nn.BCELoss()

    # This is a random vector to see the progress of the training
    random_vector = torch.randn(3, args.latent_space_size, 1, 1, device=device)

    counter = 0

    # For each epoch
    for epoch in range(args.num_epochs):

        # Array to store losses over time
        loss_generator_array = []
        loss_discriminator_array = []

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            # We train the discriminator
            discriminator.zero_grad()

            # Pass "real" samples into a tensor
            tensor = data[0].to(device)
            batch_size = tensor.size(0)

            # Perform a forward pass
            output = discriminator(tensor).view(-1)
            label = torch.full((batch_size,),1,dtype=torch.float, device=device)

            # Propagate the loss
            err_discriminator_real = criterion(output,label)
            err_discriminator_real.backward()

            # Generate "fake" samplesutput
            noise = torch.randn(batch_size,args.latent_space_size,1,1,device=device)
            fake = generator(noise)

            # Perform a forward pass
            output = discriminator(fake.detach()).view(-1)
            label = torch.full((batch_size,),0,dtype=torch.float, device=device)

            # Propagate the loss
            err_discriminator_fake = criterion(output,label)
            err_discriminator_fake.backward()

            # Optimize the discriminator
            optimizer_discriminator.step()

            # We train the generator
            generator.zero_grad()

            # Perform a forward pass
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output = discriminator(fake).view(-1)

            # Propagate the loss
            err_generator = criterion(output, label)
            err_generator.backward()

            # Optimize the generator
            optimizer_generator.step()

            # Save errors in the array
            loss_generator_array.append(err_generator.item())

            error_discriminator = err_discriminator_fake + err_discriminator_real
            loss_discriminator_array .append(error_discriminator.item())

            # Every 50 iterations we print the result of 3 random vector in the latent space to see the training
            if i % 50 == 0:
                # See the progress of the network
                print("[%d/%d] : [%d/%d]" % (epoch,args.num_epochs,i,len(dataloader)))

                # Generate three fakes samples
                output = generator(random_vector).detach().cpu().numpy()

                # Plot them in the folder output
                plt.figure()
                plt.imshow(np.transpose(output[0], (1, 2, 0)))
                plt.savefig('{}/foo{}.png'.format(args.output_dir,counter))
                plt.figure()
                plt.imshow(np.transpose(output[1], (1, 2, 0)))
                plt.savefig('{}/goo{}.png'.format(args.output_dir,counter))
                plt.figure()
                plt.imshow(np.transpose(output[2], (1, 2, 0)))
                plt.savefig('{}/hoo{}.png'.format(args.output_dir,counter))
                counter += 1

        # Save the weights for inference.py or demo.py
        if args.save_weights:

            check_point = {
                'Epoch': epoch,
                'Latent_space' : args.latent_space_size,
                'Generator' : generator.get_weights(), 
                'Discriminator': discriminator.get_weights(),
                'Loss_generator': loss_generator_array, 
                'Loss_discriminator': loss_discriminator_array,
			}

            checkpoint_name = 'check_point_epoch_{}.pth'.format(i)
            checkpoint_folder = args.checkpoints_dir

            torch.save(check_point, os.path.join(checkpoint_folder,checkpoint_name))

if __name__ == '__main__':
    main()