import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_images_dataset(path,image_size):

    # We create a transformation object to transform the image
    transformer =transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])

    # Creation of the pytorch dataset
    dataset = dset.ImageFolder(root=path,transform=transformer)
                            
    # Return the dataset
    return dataset

def get_dataloader(dataset,batch_size,shuffle=True,num_workers=2):
    return torch.utils.data.DataLoader(dataset,batch_size,shuffle=shuffle, num_workers=num_workers)