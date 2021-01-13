# import os

# images=[]

# def getFiles(path):
#     for file in os.listdir(path):
#         if file.endswith(".jpg"):
#             images.append(os.path.join(path, file))
#     return images

# images = filesPath = "nut_snacks/dataset/999999815342/"

# getFiles(filesPath)
# print(images)

import os

# specify the img directory path
path = "path/to/img/folder/"

# list files in img directory
files = os.listdir(path)

for file in files:
    # make sure file is an image
    

    if file.endswith(('.jpg', '.png ','.jpeg')):    	
        img_path = path + file

        # load file as image...