import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


import torch.nn as nn
import torch.nn.functional as F

import random


from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader, random_split


import torchvision.models as models

import pandas as pd
import numpy as np



# data set preprocess
# baseline 
# transforms = transforms.Compose(
#     [Resize((128,128)),
#     transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
                        [transforms.RandomResizedCrop(128),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4829, 0.4329, 0.3960], [0.2112, 0.1898, 0.1821])]) 
                        #DataSet's  Mean: tensor([0.4829, 0.4329, 0.3960]) Std: tensor([0.2112, 0.1898, 0.1821]) 



## hyperparameters
root = 'data/'
batch_size = 64
no_epoch = 100
model = 'resnet50'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = 'nut_snacks/dataset/'
dataset = ImageFolder(data_dir, transform=transform)
print('Total dataset images: ',len(dataset))

train_dataset, val_data = random_split(dataset, [10000, 2607])

print('train images:',len(train_dataset))
print('test images:', len(val_data))

# def train_val_dataset(dataset, val_split=0.2):
#     my_list = list(range(len(dataset)))
#     my_list = random.shuffle(my_list)
#     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size= val_split)

#     # print(list)
#     # print(my_list)
#     train_data = []
#     test_data = []
#     train_data = Subset(dataset, train_idx)
#     test_data = Subset(dataset,val_idx)

#     return train_data, test_data


# dataset = ImageFolder('nut_snacks/dataset/', transform=transforms)
# print('total images in dtaset',len(dataset))
# train_data, test_data = train_val_dataset(dataset)
# print('No of train images:',len(train_data))
# print('No of test images:',len(test_data))
# # The original dataset is available in the Subset class

train_dataloader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size,num_workers=4)
test_dataloader = torch.utils.data.DataLoader( val_data, batch_size=batch_size, shuffle=False,num_workers=4)


# train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers=4)
# x,y = next(iter(train_dataloader))
# print(x.shape, y.shape)

# test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=True, num_workers=4)
# x,y = next(iter(train_dataloader))
# print(x.shape, y.shape)






# sample model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

###--- Transfer Learning ----###
resnet50 = models.resnet50(pretrained=False)
# alexnet = models.alexnet(pretrained=False)
# squeezenet = models.squeezenet1_0(pretrained=False)
# vgg16 = models.vgg16(pretrained=False)
# densenet = models.densenet161(pretrained=False)
# inception = models.inception_v3(pretrained=False)
# googlenet = models.googlenet(pretrained=False)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=False)
# mobilenet = models.mobilenet_v2(pretrained=False)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=False)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=False)
# mnasnet = models.mnasnet1_0(pretrained=False)


net = resnet50
num_ftrs = net.fc.in_features
print('num_ftrs:',num_ftrs)

# net = net.to(device)
# define the loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)




for epoch in range(no_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # imputs = inputs.to(device)
        # labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print('oneloos')
        if i % 20 == 19:    # print every 20 mini-batches
            print('Epoch:','[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Train accuracy: %d %%' % (100 * correct / total))


print('Finished Training')


## Save our trained model

# PATH = '/checkpoints/'
# torch.save(net.state_dict(), root)


dataiter = iter(test_dataloader)
images, labels = dataiter.next()


correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # images = images.to(device)
        # labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test images: %d %%' % (
    100 * correct / total))


# Classwise accuracy of the test set
total = 0
class_correct = list(0. for i in range(258))
class_total = list(0. for i in range(258))
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        if len(labels) is batch_size:
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                total += 1
print(total)

data_info = pd.read_csv('nut_snacks/dictionary.csv', header=None)
classes = np.asarray(data_info.iloc[:, 0])

print("Classwise accuracy fo the images in test dataset")

for i in range(258):
    if class_total[i] > 0 :
        print('Accuracy of %5s : %2d %%' % (classes[i+1], 100 * class_correct[i] / class_total[i]), class_total[i])


data_dir = 'nut_snacks/dataset/'
dataset = ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=4)


# dataset = ImageFolder('nut_snacks/dataset/', transform=transforms)
# dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=4)

class_correct = list(0. for i in range(258))
class_total = list(0. for i in range(258))
total = 0
with torch.no_grad():
    for data in dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        if len(labels) is batch_size:
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                total += 1

print(total)
########### --------

########### --------
print("Classwise accuracy of the classes with less than 20 instances  in dataset:")

total_img = 0
for i in range(258):
    total_img +=  class_total[i]
    if class_total[i] > 0 and class_total[i] < 20 :
        print('Instances:', class_total[i], 'Accuracy of %12s : %2d %%' % (classes[i+1], 100 * class_correct[i] / class_total[i]), )
    
print("total images:", total_img)




## Trash 


# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Let’s load back in our saved model

# net = Net()
# net.load_state_dict(torch.load(PATH))


# outputs = net(images)


# # let’s get the index of the highest energy
# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

## Let us look at how the network performs on the whole dataset.


# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

# train_set = datasets.ImageFolder(root=root, train=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)





# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torch.utils.data as data
# import torchvision
# from torchvision import transforms

# EPOCHS = 100
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001
# TRAIN_DATA_PATH = "./images/train/"
# TEST_DATA_PATH = "./images/test/"
# TRANSFORM_IMG = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(256),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225] )
#     ])

# train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
# train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
# test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
# test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

# class CNN(nn.Module):
#     # omitted...

# if __name__ == '__main__':

#     print("Number of train samples: ", len(train_data))
#     print("Number of test samples: ", len(test_data))
#     print("Detected Classes are: ", train_data.class_to_idx) # classes are detected by folder structure

#     model = CNN()    
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     loss_func = nn.CrossEntropyLoss()    

#     # Training and Testing
#     for epoch in range(EPOCHS):        
#         for step, (x, y) in enumerate(train_data_loader):
#             b_x = Variable(x)   # batch x (image)
#             b_y = Variable(y)   # batch y (target)
#             output = model(b_x)[0]          
#             loss = loss_func(output, b_y)   
#             optimizer.zero_grad()           
#             loss.backward()                 
#             optimizer.step()

#             if step % 50 == 0:
#                 test_x = Variable(test_data_loader)
#                 test_output, last_layer = model(test_x)
#                 pred_y = torch.max(test_output, 1)[1].data.squeeze()
#                 accuracy = sum(pred_y == test_y) / float(test_y.size(0))
#                 print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)



# train
# transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# val
# transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])




# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1


# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))