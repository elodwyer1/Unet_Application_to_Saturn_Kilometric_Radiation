# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:35:03 2023

@author: eliza
"""

import numpy as np
import pandas as pd
import configparser
import tensorflow as tf
from keras import layers
import os
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']

def load_mask(id_name, path):
    #path = root + 'ML_Dataset/Main_Dataset/train'
    image_h = 384
    image_w = 128
    mask_path = os.path.join(path, id_name, "masks", id_name) + ".npy"
    _mask_image = np.load(mask_path, allow_pickle=True)
    _mask_image=np.reshape(_mask_image,(_mask_image.shape[0],_mask_image.shape[1],1))
    resize = tf.keras.Sequential([layers.Resizing(image_h, image_w),layers.Rescaling(1./255)])
    mask = resize(_mask_image)[:,:,0]
    mask=np.where(mask>0, 1, 0).reshape(image_h, image_w,1)
    return mask


##############Testing Data################
#Load Data
test_label = np.load(output_data_fp + "/test_label.npy", allow_pickle=True)
test_path = output_data_fp +  '/test'
test_results = np.load(output_data_fp + f'/{model_name}/test_results.npy', \
                  allow_pickle=True)[:,0,:,:,0]
#Probability threshold with maximum average IoU.
thresh=np.load(output_data_fp + f'/{model_name}/best_thresh.npy')
#Number of samples in test set.
test_num_ = np.arange(test_results.shape[0])
#Mask results at this threshold.
test_results_masked = np.where(test_results.copy()>=thresh, 1, 0)
#Load processed masks
test_processed_masks = np.load(output_data_fp + f'/{model_name}/test_processed_masks_withhigherthresh.npy', \
                  allow_pickle=True)
test_processed_masks = (test_processed_masks == 0)

#empty list for metrics to be appended to 
test_iou_processed_results = []
test_iou_unprocessed_results = []

count=0
#zip through each image and calculate given metric by camparing to true mask.
for i in test_num_:
    mask = load_mask(str(i).zfill(3), test_path)
    proc_result = np.ma.masked_array(test_results[i, :, :], test_processed_masks[i, :, :]).filled(0)
    iou = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=thresh)
    
    #Processed
    iou.update_state(mask, proc_result)
    test_iou_processed_results.append(iou.result().numpy())
    iou.reset_state()
    
    #Unprocessed
    iou.update_state(mask, test_results[i, :, :])
    test_iou_unprocessed_results.append(iou.result().numpy())
    iou.reset_state()
    
    print(f“test number {i}/{len(test_num_)}”, end=”\r”)
    count+=1
test_iou_processed_results = np.array(test_iou_processed_results)  
test_iou_unprocessed_results = np.array(test_iou_unprocessed_results)  
test_iou_df = pd.DataFrame({'iou': test_iou_unprocessed_results,
                            'iou_processed' :test_iou_processed_results, 'label':test_label})
#save dataframe with IoU of processed results.
test_iou_df.to_csv(output_data_fp + f'/{model_name}/test_iou_processed_withhigherthresh.csv', index=False)


########## Train############
#Load Data
train_label = np.load(output_data_fp + "/train_label.npy", allow_pickle=True)
train_path = output_data_fp +  '/train'
train_results = np.load(output_data_fp + f'/{model_name}/train_results.npy', \
                  allow_pickle=True)[:,0,:,:,0]
train_labelz = np.load(output_data_fp + "/train_label.npy", allow_pickle=True)
#Probability threshold with maximum average IoU.
#Mask results at this threshold.
train_results_masked = np.where(train_results.copy()>=thresh, 1, 0)
#Number of samples in test set.
train_num_ = np.arange(train_results.shape[0])
#load processed masks
train_processed_masks = np.load(output_data_fp + f'/{model_name}/train_processed_masks_withhigherthresh.npy', \
                  allow_pickle=True)
train_processed_masks = (train_processed_masks == 0)
train_iou_processed_results = []
train_iou_unprocessed_results = []
count=0
#zip through each image and calculate IoU by camparing to true mask.
for i in train_num_:
    mask = load_mask(str(i).zfill(3), train_path)
    result=np.ma.masked_array(train_results[i, :, :], train_processed_masks[i, :, :]).filled(0)
    #processed
    iou= tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=thresh)
    iou.update_state(mask, result)
    train_iou_processed_results.append(iou.result().numpy())
    iou.reset_state()
    #unprocessed
    iou= tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=thresh)
    iou.update_state(mask, train_results[i, :, :])
    train_iou_unprocessed_results.append(iou.result().numpy())
    iou.reset_state()
    print(count)
    print(f“test number {i}/{len(train_num_)}”, end=”\r”)
    count+=1

train_iou_processed_results = np.array(train_iou_processed_results)  
train_iou_unprocessed_results = np.array(train_iou_unprocessed_results)  
train_iou_df = pd.DataFrame({'iou': train_iou_unprocessed_results,
                             'iou_processed' :train_iou_processed_results, 'label':train_labelz})
#save IoU 
train_iou_df.to_csv(output_data_fp + f'/{model_name}/train_iou_processed_withhigherthresh.csv', index=False)

