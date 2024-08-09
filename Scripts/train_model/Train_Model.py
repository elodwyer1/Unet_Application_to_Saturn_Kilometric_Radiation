#-*- coding: utf-8 -*-


import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
from collections import defaultdict
from PIL import ImageFont
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger
from keras import layers
import visualkeras
from matplotlib.ticker import MaxNLocator
from ..read_config import config
import mlflow
import mlflow.tensorflow


"""##### Callback function for plotting accuracy per epoch."""

class PlotLossAccuracy(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.iou = []
        self.val_iou = []
        self.iou_n = []
        self.val_iou_n = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(int(self.i))

        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        self.iou.append(logs.get('iou_l'))
        self.val_iou.append(logs.get('val_iou_l'))

        self.iou_n.append(logs.get('iou_n'))
        self.val_iou_n.append(logs.get('val_iou_n'))

        self.i += 1
        
        clear_output(wait=True)
        plt.figure(figsize=(32, 7))
        av_iou = [(i+j)/2 for i, j in zip(self.val_iou_n, self.val_iou)]
        plt.subplot(141) 
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.plot(self.x, self.losses, label="train loss")
        plt.plot(self.x, self.val_losses, label="validation loss")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('loss',fontsize=25)
        plt.xlabel('epoch',fontsize=25)
        plt.title('Model Loss',fontsize=28)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize=18)

        plt.subplot(142) 
        plt.tick_params(axis='both', which='major', labelsize=12)        
        plt.plot(self.x, self.acc, label="training")
        plt.plot(self.x, self.val_acc, label="validation")
        plt.legend(fontsize=18)
        plt.ylabel('accuracy',fontsize=25)
        plt.xlabel('epoch',fontsize=25)
        plt.title('Model Accuracy',fontsize=28)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
  

        plt.subplot(143) 
        plt.tick_params(axis='both', which='major', labelsize=12)        
        plt.plot(self.x, self.iou, label="train")
        plt.plot(self.x, self.val_iou, label="validation")
        plt.plot(self.x, av_iou, label = 'average validation')
        plt.legend(fontsize=18)
        plt.ylabel('IoU',fontsize=25)
        plt.xlabel('epoch',fontsize=25)
        plt.title('IoU of LFE Class',fontsize=28)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.subplots_adjust(wspace=0.3)

        plt.subplot(144)   
        plt.tick_params(axis='both', which='major', labelsize=12)      
        plt.plot(self.x, self.iou_n, label="train")
        plt.plot(self.x, self.val_iou_n, label="validation")
        plt.plot(self.x, av_iou, label = 'average validation')
        plt.legend(fontsize=18)
        plt.ylabel('IoU',fontsize=25)
        plt.xlabel('epoch',fontsize=25)
        plt.title('IoU of Background Class',fontsize=28)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show();

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

"""##### Generator for loading label corresponding to image."""

class Label_Generator(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        label_path = os.path.join(self.path, id_name, "label", id_name) + ".npy"
        label = np.load(label_path, allow_pickle=True)
        label=str(label)
        return label


    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        if self.batch_size == 1:
          label=self.__load__(files_batch[0])
        else:
          label=[]
          for id_name in files_batch:
              label_ = self.__load__(id_name)
              label.append(label_)
        
        return label
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))


#function for each block of downsampling path.
def down_block(x, filters, do,kernel_size=(3, 3), padding='same', strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    dp1 = keras.layers.Dropout(do)(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(dp1)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p
#function for each block of up-sampling path.
def up_block(x, skip, filters, do, kernel_size=(3,3), padding='same', strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c
#function for bottleneck.
def bottleneck(x, filters, do, l1_rate, kernel_size=(3,3), padding='same', strides=1,):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", kernel_regularizer=tf.keras.regularizers.L1(l1=l1_rate))(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", kernel_regularizer=tf.keras.regularizers.L1(l1=l1_rate))(c)
    return c
#Build model.
def model(do, f, image_h, image_w, channels, l1_rate):
    inputs = keras.layers.Input((image_h, image_w, channels))
    p0 = inputs
    #Four downsampling blocks with ascending number of filters per block.
    c1, p1 = down_block(p0, f[0],do)
    c2, p2 = down_block(p1, f[1],do) 
    c3, p3 = down_block(p2, f[2],do) 
    c4, p4 = down_block(p3, f[3],do) 
    c5, p5 = down_block(p4, f[4],do)
    c6, p6 = down_block(p5, f[5],do) 
    #bottleneck
    bn = bottleneck(p6, f[6], do, l1_rate)
    
    #Four up-sampling blocks with descending number of filters per block.
    u2 = up_block(bn, c6, f[5],do)
    u3 = up_block(u2, c5, f[4],do)
    u4 = up_block(u3, c4, f[3],do) 
    u5 = up_block(u4, c3, f[2],do) 
    u6 = up_block(u5, c2, f[1],do) 
    u7 = up_block(u6, c1, f[0],do)
    

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u7)
    model = keras.models.Model(inputs, outputs)
    return model


"""##### Calculate the average of each metric for the whole training and validation set."""
'''
total_gen = DataGen(total_ids, train_path, image_h=image_h, image_w=image_w, batch_size=1)
label_gen = Label_Generator(total_ids, train_path, 1)
ac=[]
io1=[]
io2 = []
labels=[]
los=[]
train_results = []
for i in range(len(total_ids)):
  label = label_gen.__getitem__(i)
  labels.append(label)
  x,y =  total_gen.__getitem__(i)
  loss, accuracy, ioul, ioun =model.evaluate(x,y,verbose=0)
  res = model.predict(x, verbose=0)
  train_results.append(res)
  ac.append(accuracy)
  los.append(loss)
  io1.append(ioul)
  io2.append(ioun)

train_results = np.array(train_results)
np.save(checkpoint_filepath + '/train_results.npy', train_results)
train_acc_df = pd.DataFrame({'accuracy':ac, 'loss':los, 'label':labels, 'iou_LFE': io1, 'iou_background':io2})
av = lambda x, y: (x+y)/2
train_acc_df['av_iou'] = train_acc_df.apply(lambda x: av(x.iou_LFE, x.iou_background), axis=1)
train_acc_df['index']=np.arange(len(train_acc_df))
label = checkpoint_filepath +  '/train_acc_df.csv'
train_acc_df.to_csv(label)
'''
"""##### Calculate the average of each metric for the test set."""
'''
test_ac=[]
test_io1=[]
test_io2 = []
test_labels=[]
test_los=[]
test_results = []
for i in range(len(test_ids)):
  label = test_label_gen.__getitem__(i)
  test_labels.append(label)
  x,y =  test_gen.__getitem__(i)
  loss, accuracy, ioul, ioun =model.evaluate(x,y,verbose=0)
  res = model.predict(x, verbose=0)
  test_results.append(res)
  test_ac.append(accuracy)
  test_los.append(loss)
  test_io1.append(ioul)
  test_io2.append(ioun)
test_results = np.array(test_results)
np.save(checkpoint_filepath + '/test_results.npy', test_results)
test_acc_df = pd.DataFrame({'accuracy':test_ac, 'loss':test_los, 'label':test_labels, 'iou_LFE': test_io1, 'iou_background':test_io2})
av = lambda x, y: (x+y)/2
test_acc_df['av_iou'] = test_acc_df.apply(lambda x: av(x.iou_LFE, x.iou_background), axis=1)
test_acc_df['index']=np.arange(len(test_acc_df))
test_label = checkpoint_filepath +  '/test_acc_df.csv'
test_acc_df.to_csv(test_label)
'''
# Custom MLflow callback to log model weights
class MLflowModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        model_path = os.path.join(self.log_dir, f"model_epoch_{epoch}.h5")
        self.model.save_weights(model_path)
        mlflow.log_artifact(model_path)
        os.remove(model_path)  # Remove the file after logging to MLflow

### Plot schematic of model
def plot_model_schematic(model, model_name):
    font = ImageFont.truetype(config.input_data_fp + '/ARIBL0.ttf', size=60)
    color_map = defaultdict(dict)
    color_map[keras.layers.Input]['fill'] = 'slategrey'
    color_map[layers.Conv2D]['fill'] = 'steelblue'
    color_map[layers.Dropout]['fill'] = 'orange'
    color_map[layers.MaxPooling2D]['fill'] = 'powderblue'
    color_map[layers.UpSampling2D]['fill'] = 'lightgray'
    color_map[layers.Concatenate]['fill'] = 'gray'
    save_fig_to = config.output_data_fp + f'/{model_name}/figures/model_arc_vol.png'
    visualkeras.layered_view(model, legend=True,scale_xy=20,scale_z=1/9, color_map=color_map, to_file=save_fig_to,  spacing=95, font=font, draw_volume=True)# write and show


