# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:05:58 2023

@author: eliza
"""

import itertools
import cv2
import numpy as np
import pandas as pd
import configparser
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.validation import make_valid
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']


def get_contours(mask):
    cv2img = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(cv2img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    reshape_contours = lambda x: np.reshape(x, (x.shape[0], x.shape[-1]))
    reshaped_contours = [reshape_contours(i) for i in contours]
    return reshaped_contours
def map_frequencies(contour_):
    frequency = np.flip(10**(np.arange(np.log10(3.95), np.log10(1500), \
                (np.log10(1500)-np.log10(3.95))/384, dtype=float)))
    y = np.array(contour_[:,1])
    y_mapped = frequency[y]
    x=contour_[:,0]
    mapped_coords = np.array([(i,j) for i,j in zip(x, y_mapped)])
    return mapped_coords
def total_contours(vals_, results):
    im_id = []
    contours_unmapped = []
    contours_total=[]
    for i in vals_:
        mask = results[i, :, :]
        contours = get_contours(mask)
        mapped_contours = [map_frequencies(i) for i in contours]
        contours_unmapped.append(contours)
        contours_total.append(mapped_contours)
        ids_ = np.repeat(i, len(mapped_contours))        
        im_id.append(ids_)
    im_id = list(itertools.chain.from_iterable(im_id))
    contours_unmapped = list(itertools.chain.from_iterable(contours_unmapped))
    contours_total_ = list(itertools.chain.from_iterable(contours_total))
    return im_id, contours_total_, contours_unmapped
def analyse_contours(vals_, results):
    im_id, contours_total_, contours_unmapped = total_contours(vals_, results)
    ### Delta F
    fmapped_min = [min(i[:,1]) for i in contours_total_]
    min_f = []
    max_f = []
    area = []
    min_t = []
    max_t = []
    for i in contours_unmapped:
        area.append(cv2.contourArea(i))
        min_f.append(min(i[:,1]))
        max_f.append(max(i[:,1]))
        min_t.append(min(i[:,0]))
        max_t.append(max(i[:,0]))
    del_f = [(j-i) for i, j in zip(min_f, max_f)]   
    del_t = [(j-i) for i, j in zip(min_t, max_t)]  
    
    df = pd.DataFrame({'area':area,'min_f':fmapped_min, 'delta_f':del_f, 'delta_t':del_t, 'id':im_id})
    return df
### remove misclassified polygons from masks.
def remove(id_, un_df, results, c_unmapped):
    mask = results[id_, :, :].copy()
    contour_inds = un_df.loc[un_df['id']==id_, :].reset_index(drop=True)
    data_len = len(mask.flatten())
    index = np.arange(data_len, dtype=int)
    times, freqs=np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    coords = [(t, f) for t,f in zip(times.flatten(), freqs.flatten())]
    data_points = MultiPoint([Point(x, y, z) for (x, y), z in zip(coords, index)])
    #indices of each item in flattened 2D signal array.
    index = np.arange(data_len, dtype=int)
    if len(contour_inds)>0:
        inds = np.array(contour_inds.loc[:,'ind']).astype(int)
        for i in inds:
            contour = c_unmapped[i]
            if (contour.shape[0]==1):
                mask[contour[:,1], contour[:,0]] = 0
                
            elif (contour.shape[0]<3) and (contour.shape[0]>1):
                line_coords = np.array([(i, j) for i,j in zip(contour[:, 0], contour[:, 1])])
                line = LineString(line_coords)
                fund_points = line.intersection(data_points)
                fund_coords = np.array([i.coords for i in fund_points.geoms]).astype(int)[:,0,0:2]
                mask[fund_coords[:, 1], fund_coords[:, 0]] = 0
                
            else:
                fund_polygon = Polygon(contour)
                valid_shape = make_valid(fund_polygon)
                fund_points = valid_shape.intersection(data_points)
                fund_coords = np.array([i.coords for i in fund_points.geoms]).astype(int)[:,0,0:2]
                mask[fund_coords[:, 1], fund_coords[:, 0]] = 0
    
    return mask

###### Analyse Test Set ###############################
#Load Data
test_label = np.load(output_data_fp + "/test_label.npy", allow_pickle=True)
test_path = output_data_fp +  '/test'
test_results = np.load(output_data_fp + f'/{model_name}' + '/test_results.npy', \
                  allow_pickle=True)[:,0,:,:,0]
#Probability threshold with maximum average IoU.
thresh = np.load(output_data_fp + f'/{model_name}/best_thresh.npy')

######Find Contours from Masks ###########
#Mask results at this threshold.
test_results_masked = np.where(test_results.copy()>=thresh, 1, 0)
#Number of samples in test set.
test_num_ = np.arange(test_results.shape[0])
#image 'id' for loading test images.
ids = [str(i).zfill(3) for i in test_num_]
##Find coordinates of contours calculated from masks.
test_im_id, test_contours_total_, test_contours_unmapped= total_contours(test_num_, test_results_masked)
#Dataframe with the following features of each contour: area of contour, minimum frequency (in kHz),
# delta frequency (in pixels), delta time (in pixels) and the image id.
test_df = analyse_contours(test_num_, test_results_masked)
#Find the class of the image that each contour resides in.
labels_contour = test_label[test_df['id']]
#assign image class to dataframe with contour features.
test_df['label']=labels_contour
test_df.to_csv(output_data_fp + f'/{model_name}' + '/contour_analysis_test_data.csv', index=False)

######Apply post-processing criterion to contours.########
#Dataframe with information on the contours that satisfy criterion
test_selected = test_df.loc[(test_df['delta_f']>=100) & (test_df['min_f'] < 100), :]
#This is an array with the indices of contours that pass criterion with respect to total contours.
test_selected_inds = np.array(test_selected.index).astype(int)
#Assign column with index of each contour
test_selected.insert(4, 'ind', test_selected_inds, True)
#reset index of dataframe
test_selected=test_selected.reset_index(drop=True)
#Find what class of image each contour was found in
test_selected_labels = test_label[test_selected['id']]

#Find indices of unselected contours.
test_unselected_inds = [i for i in range(len(test_df)) if i not in test_selected_inds]
#Make dataframe with features describing each contour for unselected data
test_unselected = test_df.loc[test_unselected_inds, :].reset_index(drop=True)
#Assign column with index of each contour
test_unselected.insert(4, 'ind', test_unselected_inds, True)

############ Remove contours from mask that didn't fulfill criterion#########
test_processed_masks=[]
for i in test_num_:
    print(f“test number {i}/{len(test_num_)}”, end=”\r”)
    mask = remove(i, test_unselected, test_results_masked, test_contours_unmapped)   
    test_processed_masks.append(mask)
test_processed_masks=np.array(test_processed_masks)
np.save(output_data_fp + f'/{model_name}' + '/test_processed_masks_withhigherthresh.npy', test_processed_masks)


###### Analyse Training Set ###############################

#Load Data
train_label = np.load(output_data_fp + "/train_label.npy", allow_pickle=True)
train_path = output_data_fp +  '/train'
train_results = np.load(output_data_fp + f'/{model_name}' + '/train_results.npy', \
                  allow_pickle=True)[:,0,:,:,0]
train_labelz = np.load(output_data_fp + "/train_label.npy", allow_pickle=True)


######Find Contours from Masks ###########
#Mask results at this threshold.
train_results_masked = np.where(train_results.copy()>=thresh, 1, 0)
#Number of samples in test set.
train_num_ = np.arange(train_results.shape[0])
#image 'id' for loading test images.
ids = [str(i).zfill(3) for i in train_num_]
##Find coordinates of contours calculated from masks.
train_im_id, train_contours_total_, train_contours_unmapped= total_contours(train_num_, train_results_masked)
#Dataframe with the following features of each contour: area of contour, minimum frequency (in kHz),
# delta frequency (in pixels), delta time (in pixels) and the image id.
train_df = analyse_contours(train_num_, train_results_masked)
#Find the class of the image that each contour resides in.
train_labels_contour = train_labelz[train_df['id']]
#assign image class to dataframe with contour features.
train_df['label']=train_labels_contour
train_df.to_csv(output_data_fp + f'/{model_name}' + '/contour_analysis_train_data.csv', index=False)

######Apply post-processing criterion to contours.########
#Dataframe with information on the contours that satisfy criterion
train_selected = train_df.loc[(train_df['delta_f']>=100) & (train_df['min_f'] < 100), :]
#This is an array with the indices of contours that pass criterion with respect to total contours.
train_selected_inds = np.array(train_selected.index).astype(int)
#Assign column with index of each contour
train_selected.insert(4, 'ind', train_selected_inds, True)
#reset index of dataframe
train_selected=train_selected.reset_index(drop=True)
#Find what class of image each contour was found in
train_selected_labels = train_labelz[train_selected['id']]

#Find indices of unselected contours.
train_unselected_inds = [i for i in range(len(train_df)) if i not in train_selected_inds]
#Make dataframe with features describing each contour for unselected data
train_unselected = train_df.loc[train_unselected_inds, :].reset_index(drop=True)
#Assign column with index of each contour
train_unselected.insert(4, 'ind', train_unselected_inds, True)

############ Remove contours from mask that didn't fulfill criterion#########
train_processed_masks=[]
for i in train_num_:
    print(f“test number {i}/{len(train_num_)}”, end=”\r”)
    mask = remove(i, train_unselected, train_results_masked, train_contours_unmapped)   
    train_processed_masks.append(mask)
train_processed_masks=np.array(train_processed_masks)
np.save(output_data_fp + f'/{model_name}' + '/train_processed_masks_withhigherthresh.npy', train_processed_masks)
