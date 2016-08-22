'''
Title           :data_set_creation.py
Description     :This file is to create the data set and prepare this data set for vehicle - Non vehicle classification using Caffe frmework
Author          :Chitrang Talaviya
version         :0.1
python_version  :2.7.11
'''

import matplotlib.pyplot as plt
from IPython.display import Image
import os
import numpy as np
from PIL import Image
import cv2
from skimage import io
import glob
import lmdb

# Make sure that caffe is on the python path:
caffe_root = '/home/ubuntu/dhruv/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import random
import caffe
from caffe.proto import caffe_pb2


def make_datum(img, label):
    #image is  in numpy.ndarray format.
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

cur_dir = os.getcwd()
print("resizing images")
print("current directory:",cur_dir)

train_lmdb = '/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/train_lmdb'
validation_lmdb = '/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/validation_lmdb'


train_data = [img for img in glob.glob("/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/train_data/*png")]
test_data = [img for img in glob.glob("/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/test_data/*png")]

#Shuffle train_data
random.shuffle(train_data)

in_db = lmdb.open(train_lmdb, map_size=int(1e9))
#in_db = lmdb.open(train_lmdb)
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx %  6 == 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'Non' in img_path:
            label = 0
        else:
            label = 1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


print '\nCreating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e9))
#in_db = lmdb.open(validation_lmdb)
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx % 6 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'Non' in img_path:
            label = 0
        else:
            label = 1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()



