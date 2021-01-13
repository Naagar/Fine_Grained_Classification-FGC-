import pandas as pd
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pytorch_lightning as pl 

from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import Dataset, DataLoader            
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from pytorch_lightning.loggers import TensorBoardLogger






##--- Hyper parameters ---##

num_classes = 258
learning_rate = 0.001
batch_size  = 256
num_epochs = 600

# tb_logger = pl_loggers.TensorBoardLogger('logs/')
logger = TensorBoardLogger("tb_logs", name="my_model")


##--- dataset preprocess ---##

data_transforms = transforms.Compose(
    [transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

##--- dataset paths ---## 

data_dir = 'nut_snacks/dataset/'

dataset = ImageFolder(data_dir, transform=data_transforms)
print(len(dataset))

train_data, val_data = random_split(dataset, [10000, 2607])
print(len(train_data))
print(len(val_data))




##----- Selecting model for training  -----##

vgg16 = models.vgg16()
resnet18 = models.resnet18()
resnet50 = models.resnet50()

# model = vgg16
model = resnet50
# model = resnet34


class Lit_NN(pl.LightningModule):
    def __init__(self, num_classes, model):
        super(Lit_NN, self).__init__()
        self.resnet50 = model 

    def forward(self, x):
        out = self.resnet50(x)

        return out

    def training_step(self, batch, batch_idx):

        images, labels = batch
        # images = images.reshape(-1, 3*128*128)


        # Forward Pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        # acc_i = self.loss(outputs, labels)
        acc = accuracy(outputs, labels)
        pbar = {'train_acc': acc}

        tensorboard_logs = {'train_loss': loss, 'train_acc': pbar}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': pbar}
    

    def configure_optimizers(self):

        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_dataloader(self):

        
        train_dataset = train_data#seeds_dataset(train_txt_path,train_img_dir)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)


        return train_loader
    def val_dataloader(self):

        

        test_dataset = val_data #seeds_dataset(test_text_path,test_img_dir)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)


        return val_loader
    def training_step(self, batch, batch_idx):

        images, labels = batch

        outputs = self(images)
        train_loss = F.cross_entropy(outputs, labels)
        train_acc = accuracy(outputs, labels)
        tensorboard = self.logger.experiment


        self.log('train_acc',train_acc, 'train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def validation_step(self, batch, batch_idx):

        images, labels = batch
        


        # Forward Pass
        outputs = self(images)
        val_loss = F.cross_entropy(outputs, labels)
        val_acc = accuracy(outputs, labels)
        self.log('val_acc',val_acc, 'val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs = {'val_loss': val_loss, 'val_acc': acc}
        tensorboard = self.logger.experiment

        
        
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'val_acc':acc}

    def validation_epoch_ends(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        pbar = {'val_acc': acc}

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        return {'val_loss': avg_loss, 'log': tensorboard_logs,'progress_bar': pbar}


if __name__ == '__main__':

    trainer = Trainer(auto_lr_find=True, max_epochs=num_epochs, fast_dev_run=False,gpus=1, logger) # 'fast_dev_run' for checking errors, "auto_lr_find" to find the best lr_rate
    model = Lit_NN(num_classes, model)

    trainer.fit(model)



