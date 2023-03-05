# -*- coding: utf-8 -*-
import os
from os import listdir, getcwd
from os.path import join
import random
import shutil


wd = "/data2/qw/CCPD2019/ccpd10000" # 换成你所在的文件夹
detection = False

if detection:
    annotation_dir = os.path.join(wd, "labels/")
    if not os.path.isdir(annotation_dir):
        raise Exception("label dictory not found")
    image_dir = os.path.join(wd, "images/")
    if not os.path.isdir(image_dir):
        raise Exception("image dictory not found")

    train_file = open(os.path.join(wd, "2007_train.txt"), 'w')
    val_file = open(os.path.join(wd, "2007_val.txt"), 'w')
    train_file.close()
    val_file.close()

    train_file = open(os.path.join(wd, "2007_train.txt"), 'a')
    val_file = open(os.path.join(wd, "2007_val.txt"), 'a')

    list = os.listdir(image_dir) # list image files
    probo = random.randint(1, 100)
    print (len(list))
    for i in range(0, len(list)):
        path = os.path.join(image_dir,list[i])
        if os.path.isfile(path):
            image_path = image_dir + list[i]
            voc_path = list[i]
            (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
            (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
            annotation_name = nameWithoutExtention + '.txt'
            annotation_path = os.path.join(annotation_dir, annotation_name)
    #  print (annotation_path)
        probo = random.randint(1, 100)
        print("Probobility: %d" % probo)
        if(probo <= 80):
            if os.path.exists(annotation_path):
                train_file.write(image_path + '\n')
        else:
            if os.path.exists(annotation_path):
                val_file.write(image_path + '\n')
    train_file.close()
    val_file.close()
else:
    image_dir = os.path.join(wd, "images")
    list1 = os.listdir(image_dir) # list image files
    probo = random.randint(1, 100)
    print (len(list1))
    train_path = os.path.join(wd, "train")
    if not os.path.exists(train_path):
        os.mkdir(train_path) 
    val_path = os.path.join(wd, "val")
    if not os.path.exists(val_path):
        os.mkdir(val_path) 
    for i in range(0, len(list1)):
        path = os.path.join(image_dir,list1[i])
        if os.path.isfile(path):
            image_path = os.path.join(image_dir , list1[i])
    #  print (annotation_path)
        probo = random.randint(1, 100)
        print("Probobility: %d" % probo)
        if(probo <= 80):
            shutil.copy(path,os.path.join(train_path,list1[i]))
        else:
            shutil.copy(path,os.path.join(val_path,list1[i]))

