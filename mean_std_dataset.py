import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
                         # load the data Set

from torch.utils.data import  random_split
from torchvision.datasets import ImageFolder


batch_size = 256
data_dir = 'nut_snacks/dataset/'

data_transforms = transforms.Compose(
                        [transforms.RandomResizedCrop(128),
                        
                        transforms.ToTensor(),
                        ])

dataset = ImageFolder(data_dir, transform=data_transforms)
print('Total dataset images: ',len(dataset))


loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size)




def mean_std(loader):
	mean = 0
	std = 0
	for images, _ in loader :
		batch_samples = images.size(0)
		images = images.view(batch_samples, images.size(1), -1)
		mean += images.mean(2).sum(0)
		std += images.std(2).sum(0)
	mean /= len(loader.dataset)
	std /= len(loader.dataset)  
	return mean,std

mean, std = mean_std(loader)

print(f'Mean: {mean}')

print(f'Std: {std}')
