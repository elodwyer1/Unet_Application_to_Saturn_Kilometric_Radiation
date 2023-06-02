# -*- coding: utf-8 -*-
"""##### Import Modules"""

import zipfile
import os
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'

## Seeding 
seed = 42
random.seed = seed
np.random.seed = seed
tf.seed = seed

"""#### Data Generator 
Loads training data from files in batches according to batch_size.
Data is stored in the form of a single folder with an 'id' for each instance of training data i.e a single folder containing input data (spectrograms of flux density and normalised degree of circular polarization and values of latitude and local time median and standard deviation over which the spectrogram was recorded) and label (true mask and LFE class). 
Example is folder with name '000' containing subfolders 'images', 'traj','label' and 'masks'. 

*   'images' contains file '000.npy' containing a 3D array with the spectrograms showing flux density and normalised degree of circular polarization contained in one image.
*   'traj' contains file '000.npy', an array with 4 values corresponding to the latitude standard deviation, latitude median, local time standard deviation and local time median.
* 'label' contains file '000.npy', an array with a single value. It is a string denoting the class of the image. May be 'LFE', 'LFE_sp', 'LFE_m', 'LFE_ext', 'LFE_dg' or 'NoLFE'.
* 'masks' contains file '00.npy' with a 2D array showing the corresponding labelled mask for the spectrograms of flux density and normalised degree of circular polarization.

The data generator returns input data and correspond true mask when the '__getitem__' attribute is called. The input data is of shape (batch size, 1, image height, image width, number of channels) and the output data is of shape (batch size, 1, image height, image width, 1). 
"""

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size, image_w, image_h):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_w = image_w
        self.image_h =image_h
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ###########Path###############
        image_path = os.path.join(self.path, id_name, "images", id_name) + ".npy"
        mask_path = os.path.join(self.path, id_name, "masks", id_name) + ".npy"

        #######Reading Image###########
        #load 3D array of flux density and normalised degree of circular polarization.
        image = np.load(image_path, allow_pickle=True)
        #Resize array to pre-defined width and height using bilinear interpolation.
        resize = tf.keras.Sequential([layers.Resizing(self.image_h, self.image_w)])
        image = resize(image)

        ##########Reading Masks##########
        #Load 2D array with labellled mask corresponding to 3D array above.
        mask_image = np.load(mask_path,allow_pickle=True)
        mask_image=np.reshape(mask_image,(mask_image.shape[0], mask_image.shape[1],1))
        #Resize array to pre-defined width and height using bilinear interpolation.
        resize = tf.keras.Sequential([layers.Resizing(self.image_h, self.image_w)])
        mask = resize(mask_image)[:,:,0]
        #Ensure mask still consists of only ones and zeroes after interpolation.
        mask=np.where(mask>0, 1, 0).reshape(self.image_h,self.image_w,1)

        ###########Label############
        #Load class label for image.
        label_path = os.path.join(self.path, id_name, "label", id_name) + ".npy"
        label = np.load(label_path, allow_pickle=True)

        #########Trajectory##########
        traj_path = os.path.join(self.path, id_name, "traj", id_name) + ".npy"
        #Load array containing trajectory data.
        traj = np.load(traj_path, allow_pickle=True)
        #Define Latitude standard deviation, Latitude median, Local Time standard deviation and Local Time median.
        lat_s, lat_m, lt_s, lt_m = traj[0], traj[1], traj[2], traj[3]
        #Define 2D array of repeating values in the same width and height as image and mask data for each trajectory value.
        lat_s_arr=np.full((self.image_h, self.image_w, 1),lat_s)
        lat_m_arr=np.full((self.image_h, self.image_w, 1),lat_m)
        lt_s_arr =np.full((self.image_h, self.image_w, 1),lt_s)
        lt_m_arr=np.full((self.image_h, self.image_w, 1),lt_m)

        ###########Frequency Channels #########
        #Define 2D array of 384 evenly spaced values between 0 and 1 repeating at each step in time for same dimension as 2D arrays above.
        step = 1/383
        f=np.arange(0, 1+step, step)
        f_ = np.repeat(f, 128).reshape(len(f), 128, 1)

        #Concatenate each array to one 3D array of all input data of shape (image_h, image_w, number of channels)
        im_all_channels = np.concatenate([image, lat_s_arr, lat_m_arr,lt_s_arr, lt_m_arr, f_], axis=2)

        return im_all_channels, mask


    def __getitem__(self, index):
      #this generates input and output data for each batch.
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        im=[]
        mask=[]
        
        for id_name in files_batch:
            _im_all_channels, _mask = self.__load__(id_name)
            im.append(_im_all_channels)
            mask.append(_mask)
        
        im=np.array(im)
        mask=np.array(mask)
        return im, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

"""#### Load Data from Drive"""

#Load training data from drive and unzip into colab notebook.
#path to train data in colab notebook.
train_path = output_data_fp + "/train/"
##Load Ids
random_ids = next(os.walk(train_path))[1]
count=3000
ids=[str(i).zfill(3) for i in np.arange(count)]
total_ids = [i for i in ids if i in random_ids]
# Separate into training and validation
#Train set is 75% of total, and validation is 1/3 of the train set and so it is 25% of total data
train_label=np.load(output_data_fp + '/train_label.npy',allow_pickle=True)
train_ids, valid_ids = train_test_split(total_ids, test_size=.35, random_state=42,stratify=train_label)
#path to test data in colab notebook.
test_path= output_data_fp + '/test'
## Load Testing Ids
test_ids = next(os.walk(test_path))[1]
string_ids = [str(i).zfill(3) for i in range(1000)]
test_ids = [i for i in string_ids if i in test_ids]
#Define image dimensions
image_h = 384
image_w = 128
#Initiate instances of data generator for the training and testing data.
train_gen = DataGen(total_ids, train_path, image_h=image_h, image_w=image_w, batch_size=1)
test_gen = DataGen(test_ids, test_path, image_h=image_h, image_w=image_w, batch_size=1)

"""##### Load pre-trained model."""

#Name of file containing model
checkpoint_filepath = output_data_fp + f'/{model_name}'
model = keras.models.load_model(checkpoint_filepath)

"""##### Load results of predicted masks from train and test set from pre-trained model."""

train_results = np.load(checkpoint_filepath+'/train_results.npy', allow_pickle=True)
test_results = np.load(checkpoint_filepath+'/test_results.npy', allow_pickle=True)
train_acc_df = pd.read_csv(checkpoint_filepath + '/train_acc_df.csv')
test_acc_df = pd.read_csv(checkpoint_filepath + '/test_acc_df.csv')
test_labels = test_acc_df['label']
test_labels_s = ['LFE' if i!='NoLFE' else i for i in test_labels]

"""##### Calculate IoU per image in test set for range of thresholds."""

thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

iou_total = []
test_num = np.arange(len(test_ids))
for thresh in thresholds:
    print(thresh)
    count=0
    ious = []
    for i in test_num:
        result = test_results[i,:,:]
        x, y = test_gen.__getitem__(i)
        mask = y[0,:,:,0]
        iou= tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=thresh)
        iou.update_state(mask, result)
        ious.append(iou.result().numpy())
        iou.reset_state()
        count+=1
    ious=np.array(ious).reshape((len(ious), 1))
    iou_total.append(ious)
      
iou_total_arr = np.concatenate(iou_total, axis=1)
np.save(checkpoint_filepath + '/test_iou_perthresh.npy', iou_total_arr)

"""##### Plot average IoU at each threshold."""

#Calculate average IoU of all images for each threshold.
avs = [np.mean(iou_total_arr[:, i]) for i in range(iou_total_arr.shape[1])]
avs = [round(i, 3) for i in avs]
ind_max = avs.index(max(avs))
np.save(checkpoint_filepath + '/best_thresh.npy', np.array(thresholds[ind_max]))
#Plot average IoU vs threshold.
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
plt.plot(thresholds, avs)
plt.scatter(thresholds, avs)
best_iou = max(avs)
best_iou = round(best_iou, 2)
ax.vlines(thresholds[ind_max], 0, 1, linestyle='dotted')
ax.set_ylim(0.4, 1)
ax.set_title('IoU vs Threshold for Test Set')
ax.set_xlabel('Threshold')
ax.set_ylabel('Average IoU')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
plt.savefig(checkpoint_filepath + '/threshold_vs_iou.png')
plt.show()

