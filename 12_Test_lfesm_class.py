# -*- coding: utf-8 -*-


"""##### Import Modules"""
import zipfile
import os
import random
import numpy as np
import pandas as pd
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
test_path= output_data_fp + "/lfe_sm/test_lfesm/"
## Load Testing Ids
test_ids = next(os.walk(test_path))[1]
string_ids = [str(i).zfill(3) for i in range(1000)]
test_ids = [i for i in string_ids if i in test_ids]
print(len(test_ids))
#Define image dimensions
image_h = 384
image_w = 128
#Initiate instances of data generator for the training and testing data.
test_gen = DataGen(test_ids, test_path, image_h=image_h, image_w=image_w, batch_size=1)

#Path to folder where model will be stored.
checkpoint_filepath = output_data_fp + model_name
thresh = np.load(checkpoint_filepath + '/best_thresh.npy')

"""##### Load pre-trained model."""

model = keras.models.load_model(checkpoint_filepath)

"""### Test"""

test_ac=[]
test_iou=[]
test_labels=[]
test_los=[]
test_results = []
for i in range(len(test_ids)):
  print(i)
  x,y =  test_gen.__getitem__(i)
  loss, accuracy, ioul, ioun =model.evaluate(x,y,verbose=0)
  res = model.predict(x, verbose=0)
  test_results.append(res)
  test_ac.append(accuracy)
  test_los.append(loss)
  iou = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=thresh)
  iou.update_state(y, res)
  test_iou.append(iou.result().numpy())
  iou.reset_state()
    
test_results = np.array(test_results)
np.save(checkpoint_filepath + '/test_lfesm_results.npy', test_results)
test_acc_df = pd.DataFrame({'accuracy':test_ac, 'loss':test_los, 'iou': test_iou})
test_acc_df.to_csv(checkpoint_filepath + '/lfe_sm_metrics.csv', index=False)