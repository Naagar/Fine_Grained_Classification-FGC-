# Fine Grained Classification

## Project report 

Click [here](https://www.overleaf.com/read/hybjpqspktym) or download the pdf(Fine Grained Classification Project Report.pdf) above


## Dataset
To download dataset follow the instructions [here](). A nice script that simplifies downloading and extracting can be found here: 

  The dataset comprises images from the nut snacks category. Several aspects that makes recognition of this category particularly interesting are:

      Distribution of samples per class is highly skewed
      More often than not, the discerning feature is either localized text or dimensions of the object
      Conditions under which the photo was captured may not always be optimal
      Occlusion by other objects and promotional signs.


train and test split 80:20

  10,000   :  2,607



| Loss | Accuracy |
| --- | --- |
| ![bnaf_u1](FGC_results/train_loss.png) | ![](FGC_results/val_loss.png) |
| ![bnaf_u2](FGC_results/train_acc_top1.png) | ![](FGC_results/test_acc_top5.png) |
| ![bnaf_u3](FGC_results/test_acc_top1.png) | ![bnaf_2spirals](FGC_results/val_loss.png) |
| ![bnaf_u4](FGC_results/best_test_acc.png) | ![bnaf_rings](FGC_results/test_acc.png) |


### mean and std of the data set, run
```
python mean_std_dataset.py

```


#### Usage

#### Results
I trained two models:

### To run the base model, run

```
python fine_grained.py
```

To train a model using pytorch package:
```
python main_fg_2.py --data_dir [path to data source] \
               --epochs=100 \
               --arch [model name eg. resnet18, resnet50,googlenet,squeezenet] \
               --input_img_size= 128 \
               --lr=0.1
               --batch_size=64 [this is per GPU]
```
To use the pretrained model
To train a model using pytorch package:
```
python main_fg_2.py --pretrained \
                --data_dir [path to data source] \
               --epochs=100 \
               --arch [model name eg. resnet18, resnet50,googlenet,squeezenet(from https://pytorch.org/docs/stable/torchvision/models.html)] \
               --input_img_size= 128 \
               --lr=0.1
               --batch_size=64 [this is per GPU]
```



To evaluate model:
```
python main_fg_2.py --evaluate \
               --restore_file=[path to .pt checkpoint] \
               --data_dir=[path to data source] \
               --[options of the saved model: n_levels, depth, width, batch_size]
```


## pyTorchlightning implimentation run

```
python main_pl.py 
```

## To run the tensorboard 

```
tensorboard --logdir=runs_FGC

```



#### References
* Official implementation 




## Dependencies
* python 3.6
* pytorchlighting
* pytorch 
* numpy
* matplotlib
* tensorboard
* cuda

