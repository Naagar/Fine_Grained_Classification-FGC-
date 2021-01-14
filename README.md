# Fine Grained Classification

## Project report 

Click [here](https://www.overleaf.com/read/hybjpqspktym) or download the pdf(Fine Grained Classification Project Report.pdf) above


## Dataset
To download dataset follow the instructions [here](). A nice script that simplifies downloading and extracting can be found here: 



train and test split 80:20

  10,000   2,607



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

