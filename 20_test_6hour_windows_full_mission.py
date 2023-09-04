# -*- coding: utf-8 -*-


"""##### Import Modules"""
import os
#import zipfile_deflate64 as zipfile
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import math
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

        #######Reading Image###########
        #load 3D array of flux density and normalised degree of circular polarization.
        image = np.load(image_path, allow_pickle=True)
        #Resize array to pre-defined width and height using bilinear interpolation.
        resize = tf.keras.Sequential([layers.Resizing(self.image_h, self.image_w)])
        image = resize(image)



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

        return im_all_channels


    def __getitem__(self, index):
      #this generates input and output data for each batch.
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        im=[]


        for id_name in files_batch:
            _im_all_channels= self.__load__(id_name)
            im.append(_im_all_channels)

        im=np.array(im)

        return im

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
#function for averaging the overlapping parts of images
def overlap_corners(x, y, n):
  lefts = []
  rights = []
  left = x+y
  right = left +2*y
  lefts.append(left)
  rights.append(right)
  for i in range(2, n):
    left  = right+x
    right = left+2*y
    lefts.append(left)
    rights.append(right)
  return lefts, rights
def middles(x, y, n):
  lefts = []
  rights = []
  left = 0
  right = x+y
  lefts.append(left)
  rights.append(right)
  for i in range(2, n):
    left= right+2*y
    right=left+x
    lefts.append(left)
    rights.append(right)
  lefts.append(right+2*y)
  rights.append(left+x+y)
  return lefts, rights
def get_overlaps(x, y, n, test):
  left_overlap, right_overlap = overlap_corners(x, y, n)
  overlaps = [[i, j] for i, j  in zip(left_overlap, right_overlap)]
  overlap_pairs = []
  for i in range(n-1):
    o = overlaps[i]
    o = test[:, o[0]:o[1]]
    o_ = np.array(np.hsplit(o.copy(), 2))
    overlap_pairs.append(o_)
  overlap_pairs = np.array(overlap_pairs)
  return overlap_pairs
def av_overlap_combined(x, y, n, test):
  left_overlap, right_overlap = overlap_corners(x, y, n)
  left_middle, right_middle = middles(x, y, n)
  overlaps = [[i, j] for i, j  in zip(left_overlap, right_overlap)]
  middles_ = [[i, j] for i, j in zip(left_middle, right_middle)]
  total_images = []
  for i in range(n-1):
    m = middles_[i]
    o = overlaps[i]
    middle= test[:, m[0]:m[1]]
    o = test[:, o[0]:o[1]]
    o_ = np.array(np.hsplit(o.copy(), 2))
    av_o = np.mean(o_, axis=0)
    final_arr = np.concatenate([middle, av_o], axis=1)

    total_images.append(final_arr)
  #last_im = np.concatenate([av_o,test[:,middles_[n-1][0]:]], axis=1)

  total_images.append(test[:,middles_[n-1][0]:])
  total_images = np.concatenate(total_images, axis=1)
  return total_images

"""#### Load Data from Drive"""

data_name = '2004001_2017258'
test_path= output_data_fp + f'/data_{data_name}/test_{data_name}/'
## Load Testing Ids
test_ids = next(os.walk(test_path))[1]
string_ids = [str(i).zfill(4) for i in range(40000)]
test_ids = [i for i in string_ids if i in test_ids]
#Define image dimensions
image_h = 384
image_w = 128
#Initiate instances of data generator for the training and testing data.
test_gen = DataGen(test_ids, test_path, image_h=image_h, image_w=image_w, batch_size=1)

"""##### Load pre-trained model."""
checkpoint_filepath = output_data_fp + f'/{model_name}'
model = keras.models.load_model(checkpoint_filepath)

#value used for thresholding mask
best_test_iou = np.load(checkpoint_filepath+'/best_thresh.npy')
"""### Test"""

#save results to folder after generating prediction
result_path = checkpoint_filepath+ f'/{data_name}/results/'
if not os.path.exists(result_path):
  os.makedirs(result_path)
num_saved = len(next(os.walk(checkpoint_filepath+ f'/{data_name}/results')))
test_results = []
for i in range(num_saved, len(test_ids)):
  print(f'{round(100 * i/len(test_ids),4)} %')
  x =  test_gen.__getitem__(i)
  res = model.predict(x, verbose=0)
  np.save(result_path +  f'{str(i).zfill(5)}.npy', res)

## Load Testing Ids
result_ids = next(os.walk(result_path))[2]
result_string_ids = [str(i).zfill(5)+'.npy' for i in range(40000)]
result_ids = [i for i in result_string_ids if i in result_ids]

#Load and join together predictions in two halves.
num_images = 32923
first_half = math.ceil(num_images/2)
ids_first = np.arange(first_half)
second_half = math.floor(num_images/2)
ids_second = np.arange(first_half, first_half + second_half, 1)

test_results = []
for i in ids_first:
  print(f'{round(100 * i/(first_half-1),4)} %')
  r = np.load(result_path + f'/{str(i).zfill(5)}.npy')
  test_results.append(r)
test_results = np.array(test_results)
np.save(checkpoint_filepath + f'/{data_name}_1.npy', test_results)

test_results = []
for i in ids_second:
    print(f'{round(100 * i/(second_half-1),4)} %')
    r = np.load(result_path + f'/{str(i).zfill(5)}.npy')
    test_results.append(r)
test_results = np.array(test_results)
np.save(checkpoint_filepath + f'/{data_name}_2.npy', test_results)

"""### Get Average overlap of images."""
test_results1 = np.load(checkpoint_filepath + f'/{data_name}_1.npy')
test_results2 = np.load(checkpoint_filepath + f'/{data_name}_2.npy')

#concatenate to get total of the predicted images in one numpy array.
test_results = np.concatenate([test_results1, test_results2],axis=0)
#concatenate results into one long array.
test_results_concat = np.concatenate(test_results[:,0,:,:,0], axis=1)
#save array in this format.
np.save(checkpoint_filepath + f'/{data_name}_total_results.npy', test_results_concat)


#Average overlaps and rejoin
#test_results_concat = np.load(checkpoint_filepath + f'/{data_name}_total_results.npy', allow_pickle=True)
test_results_av_overlap_comb = av_overlap_combined(18, 55, len(test_results), test_results_concat)
np.save(checkpoint_filepath + f'/{data_name}_av_overlap_combined.npy', test_results_av_overlap_comb)
