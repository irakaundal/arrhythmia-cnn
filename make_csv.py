from utils import BASE_DIR
import pandas as pd
import random
import csv
import os
import math
import numpy as np

obj = {}
validation_csv = [['image_name', 'class']]
test_csv = [['image_name', 'class']]
sample_dir = os.path.join(BASE_DIR, 'samples')
list_samples = os.listdir(sample_dir)

with open('eval.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = row['image_name']
        y = row['key']
        if y == 'key':
            continue
        if y in obj:
            obj[y]+=1
        else:
            obj[y]=1
        if obj[y] <=1400:
            validation_csv.append([x,y])
        else:
            test_csv.append([x,y])

with open('validation.csv','w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(validation_csv)
with open('test.csv','w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(test_csv)


'''for i in range(0,len(list_samples)):
        _class  = list_samples[i]
        class_dir = os.path.join(sample_dir,_class)
        list_images = os.listdir(class_dir)
        temp_list_images = list(list_images)
        print(_class,len(list_images))
        n = len(list_images)
        for i in range(0, math.floor(8000/n)):
            temp_list_images.extend(list_images)
        if _class not in obj:
            obj[_class] = temp_list_images
for key, pair in obj.items():
    total  = len(pair)
    print(key,total)
    obj[key] = random.sample(pair,8000)
trainList = [['image_name', 'key']]
testList = [['image_name', 'key']]
for key, pair in obj.items():
    total  = len(pair)
    print(key,total)
    for i  in range(0,math.floor(0.8*len(pair))):
        trainList.append([pair[i],key])
    for i  in range(math.floor(0.8*len(pair)), len(pair)):
        testList.append([pair[i],key])
random.shuffle(trainList)
random.shuffle(testList)
with open('train.csv','w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(trainList)
with open('test.csv','w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(testList)

df = pd.read_csv('train.csv', header=None)
ds = df.sample(frac=1)
ds.to_csv('train_new.csv')
df = pd.read_csv('test.csv', header=None)
ds = df.sample(frac=1)
ds.to_csv('test_new.csv')'''