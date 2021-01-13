import cv2 
import os 
import glob 
import pandas as pd


img_dir = "nut_snacks/dataset/"   
data_path = os.path.join(img_dir,'*g') 
files = glob.glob('nut_snacks/dataset/') 
data = [] 
df_train= pd.DataFrame(columns=['img_name', 'class_no'])
i_train = 0

print(files)
for f1 in files: 
    img = cv2.imread(f1) 
    # validation_counter += 1
    # df_train.loc[i_train] = f1 , label
    i_train += 1
    data.append(img)
print(i_train)
print("finish")
df_train.to_csv('data_file.csv')              

