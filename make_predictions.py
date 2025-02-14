'''
Title           :make_predictions.py
Description     :This file is for making predictions on the new data.
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
import time

# Make sure that caffe is on the python path:
caffe_root = '/home/ubuntu/dhruv/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import random
import caffe
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    #img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/caffe_model_1/caffenet_deploy_1.prototxt',
                '/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/caffe_model_1_iter_5000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/test_data/*png")]
start = time.time()
total = 0
#Making predictions
test_ids = []
preds = []
for img_path in test_img_paths:
    total = total + 1
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probas.argmax()]

    print img_path
    print pred_probas.argmax()
    print '-------'

end = time.time()

time_taken = end - start
print "Time Taken: " + str(time_taken)
print "Total images for testing is: " + str(total)
total = time_taken / total 
print "Time taken for single image is: " + str(total)
'''
Making submission file
'''
with open("/home/ubuntu/Documents/chitrang/Deep_Learning_Car_295B/caffe_model_1/submission_model_1.csv","w") as f:
    f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i])+","+str(preds[i])+"\n")
f.close()
