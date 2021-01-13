# Fine Grained Classification


## Dataset
To download dataset follow the instructions [here](). A nice script that simplifies downloading and extracting can be found here: 

#### Results
I trained two models:

### mean and std of the data set, run
```
python mean_std_dataset.py

```
### To run the base model, run

```
python fine_grained.py
```


##### Model A attribute manipulation on in-distribution sample:

Embedding vectors were calculated for the first 30K training images and positive / negative attributes were averaged then subtracting. The resulting `dz` was ranged and applied on a test set image (middle image represents the unchanged / actual data point).



##### Model A attribute manipulation on 'out-of-distribution' sample (i.e. me):


#### Usage

To train a model using pytorch package:
```
python main_fg_2.py --train \
               --distributed \
               --dataset=celeba \
               --data_dir=[path to data source] \
               --n_levels=3 \
               --depth=32 \
               --width=512 \
               --batch_size=16 [this is per GPU]
```

To evaluate model:
```
python main_fg_2.py --evaluate \
               --restore_file=[path to .pt checkpoint] \
               --dataset=celeba \
               --data_dir=[path to data source] \
               --[options of the saved model: n_levels, depth, width, batch_size]
```


## pyTorchlightning implimentation run

```
python main_pl.py 
```


#### Datasets

To download CelebA follow the instructions [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). A nice script that simplifies downloading and extracting can be found here: https://github.com/nperraud/download-celebA-HQ/


#### References
* Official implementation 




## Dependencies
* python 3.6
* pytorchlighting
* pytorch 
* numpy
* matplotlib
* tensorboard


