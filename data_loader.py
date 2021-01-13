import torch

import random


from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader


def train_val_dataset(dataset, val_split=0.2):
	my_list = list(range(len(dataset)))
	my_list = random.shuffle(my_list)
	train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size= val_split)

	print(list)
	print(my_list)
	train_data = []
	test_data = []
	train_data = Subset(dataset, train_idx)
	test_data = Subset(dataset,val_idx)

	return train_data, test_data
# def train_val_dataset(dataset, val_split=0.20):

# 	my_list = list(range(len(dataset)))
# 	my_list = random.shuffle(my_list)
#     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
#     print(list,my_list)
#     train_data = 0
#     test_data = 0
#     train_data = Subset(dataset, train_idx)
#     test_data = Subset(dataset, val_idx)

#     return train_data, test_data

dataset = ImageFolder('nut_snacks/dataset/', transform=Compose([Resize((224,224)),ToTensor()]))
print(len(dataset))
train_data, test_data = train_val_dataset(dataset)
print(len(train_data))
print(len(test_data))
# The original dataset is available in the Subset class
# print(datasets['train'].dataset)

train_dataloaders = DataLoader(train_data, batch_size = 32, shuffle=True, num_workers=4)
x,y = next(iter(train_dataloaders))
print(x.shape, y.shape)

test_dataloaders = DataLoader(test_data, batch_size = 32, shuffle=True, num_workers=4)
x,y = next(iter(train_dataloaders))
print(x.shape, y.shape)



##
# def train_val_dataset(dataset, val_split=0.20):

# 	my_list = list(range(len(dataset)))
# 	my_list = random.shuffle(my_list)
#     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
#     print(list,my_list)
#     train_data = 0
#     test_data = 0
#     train_data = Subset(dataset, train_idx)
#     test_data = Subset(dataset, val_idx)

#     return train_data, test_data