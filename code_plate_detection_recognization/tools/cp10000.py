import os 
import shutil
import random


images_dir = "/data2/qw/CCPD2019/images" # 换成放入图片的八万多张数据
new_dir ="/data2/qw/CCPD2019/ccpd10000/images/" # 复制到新的文件夹下面

n = len(os.listdir(images_dir))
random.shuffle(os.listdir(images_dir))
for i in range(10000):
    image_path = os.path.join(images_dir, os.listdir(images_dir)[i]) 
    shutil.copy(image_path,os.path.join(new_dir, os.listdir(images_dir)[i]))
    label_name = os.listdir(images_dir)[i].replace(".jpg",".txt")
    label_path =image_path.replace(".jpg",".txt").replace("images","labels")
    shutil.copy(label_path,os.path.join(new_dir.replace("images","labels"), label_name))
    print(i)