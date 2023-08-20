# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:05:36 2023

@author: eliza
"""
import numpy as np
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
import time as t
import pandas as pd
from os import path
from tfcat import TFCat
import shapely.geometry as sg
import configparser
import tensorflow as tf
from shapely.validation import make_valid
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'

def legend_without_duplicate_labels(ax, loc_):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc_,fontsize=14)

'''____This was adapted from original code for the space labelling tool!!!___'''
def get_polygons(polygon_fp,start, end):
    unix_start=t.mktime(start.utctimetuple())
    unix_end=t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array=[]
    if path.exists(polygon_fp):
        catalogue = TFCat.from_file(polygon_fp)
        for i in range(len(catalogue)):
            time_points=np.array(catalogue._data.features[i]['geometry']['coordinates'][0])[:,0]
            if any(time_points <= unix_end) and any(time_points >= unix_start):
                
                polygon_array.append(np.array(catalogue._data.features[i]['geometry']['coordinates'][0]))
#polgyon array contains a list of the co-ordinates for each polygon within the time interval         
    return polygon_array

'''____This was adapted from original code for the space labelling tool!!!___'''
def extract_data(time_view_start, time_view_end, val):
    # read the save file and copy variables
    time_view = time.copy()[(time >= time_view_start) & (time < time_view_end)]
    # copy the flux and frequency variable into temporary variable in
    # order to interpolate them in log scale
    if val == 's':
        #print('s')
        s = flux.copy()[:, (time >= time_view_start) & (time < time_view_end)]
    elif val == 'v':
        #print('v')
        s = pol.copy()[:, (time >= time_view_start) & (time < time_view_end)]
   
    frequency_tmp = np.load(output_data_fp + '/frequency.npy', allow_pickle=True).copy()
    # frequency_tmp is in log scale from f[0]=3.9548001 to f[24] = 349.6542
    # and then in linear scale above so it's needed to transfrom the frequency
    # table in a full log table and einterpolate the flux table (s --> flux
    step = (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/383
    frequency = 10**(np.arange(np.log10(frequency_tmp[0]), np.log10(frequency_tmp[-1])+step/2, step, dtype=float))
    flux_ = np.zeros((frequency.size, len(time_view)), dtype=float)
    for i in range(len(time_view)):
        flux_[:, i] = np.interp(frequency, frequency_tmp, s[:, i])
    return time_view, frequency, flux_

def find_mask(time_view_start, time_view_end, val, polygon_fp):
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval
    polygon_array=get_polygons(polygon_fp, time_view_start, time_view_end)
    #signal data and time frequency values within the time range specified.
    time_dt64, frequency, s=extract_data(time_view_start, time_view_end, val)
    time_unix=[i.astype('uint64').astype('uint32') for i in time_dt64]
    times, freqs=np.meshgrid(time_unix, frequency)
    #Total length of 2D signal array.
    data_len = len(s.flatten())
    #indices of each item in flattened 2D signal array.
    index = np.arange(data_len, dtype=int)
    #Co-ordinates of each item in 2D signal array.
    coords = [(t, f) for t,f in zip(times.flatten(), freqs.flatten())]
    data_points = sg.MultiPoint([sg.Point(x, y, z) for (x, y), z in zip(coords, index)])
    #Make mask array.
    mask = np.zeros((data_len,))
    #Find overlap between polygons and signal array.
    #Set points of overlap to 1.
    for i in polygon_array:
        fund_polygon = sg.Polygon(i)
        valid_shape = make_valid(fund_polygon)
        fund_points = valid_shape.intersection(data_points)
        if len(fund_points.bounds)>0:
            mask[[int(geom.z) for geom in fund_points.geoms]] = 1
    mask = (mask == 0)
    
    #Set non-polygon values to zero in the signal array.
    s = np.nan_to_num(s)
    flux_ones = np.where(s>0, 1, 0)
    v = np.ma.masked_array(flux_ones, mask=mask).filled(0)
    
    
    return v

#load in full dataset
print('load data')
print('time')
t_timestamp = np.load(input_data_fp + "/time_indatetime_all_years.npy", allow_pickle=True)
t_isostring = np.array([datetime.strftime(i,'%Y-%m-%dT%H:%M:%S') for i in t_timestamp])
time =t_isostring
time = np.array(time, dtype=np.datetime64)
print('flux')
flux = np.load(input_data_fp + "/s_all_years.npy", allow_pickle=True)
print('polarization')
pol = np.load(input_data_fp + "/p_all_years.npy", allow_pickle=True)


#find true and predicted mask for data
data_name = 'test_2006'
#start end end dates that we are looking at
start = pd.Timestamp('20060101')
end = pd.Timestamp('2007-01-01T02:45:00')
t_start = np.arange(start, end, np.timedelta64(384, "m"))
t_end = np.array([i+np.timedelta64(384, "m") for i in t_start])
t_start=t_start.astype(datetime)
t_end=t_end.astype(datetime)
year = datetime.strftime(start, '%Y')
#polygon filepaths
true_poly_fp = input_data_fp + '/SKR_LFEs.json'
predicted_poly_fp = output_data_fp + f'/{model_name}/{data_name}_selected_contours.json'

#calculate true mask for full year.
true_masks=[]
for day1, day2 in zip(t_start, t_end):
    print(day1)
    a=find_mask(day1, day2, 'v', true_poly_fp) 
    true_masks.append(a)
#true_masks=np.concatenate(true_masks)
true_masks=np.array(true_masks)
np.save(output_data_fp + f'/data_{year}/full_{year}_true_mask.npy', true_masks)

##calculate predicted mask for full year
predicted_masks=[]
for day1, day2 in zip(t_start, t_end):
    print(day1)
    a=find_mask(day1, day2, 'v', predicted_poly_fp) 
    predicted_masks.append(a)
#predicted_masks=np.concatenate(predicted_masks)
predicted_masks=np.array(predicted_masks)
np.save(output_data_fp + f'/{model_name}/full_{year}_predicted_mask.npy', predicted_masks)

#reload to save memory
year=2006
true_mask = np.load(output_data_fp + f'/data_{year}/full_{year}_true_mask.npy')
true_mask = np.concatenate(true_mask, axis=1)
predicted_mask = np.load(output_data_fp + f'/{model_name}/full_{year}_predicted_mask.npy')
predicted_mask = np.concatenate(predicted_mask, axis=1)

#calculate iou
thresh=np.load(output_data_fp +f'/{model_name}/best_thresh.npy')
iou= tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=thresh)
iou.update_state(true_mask, predicted_mask)
iou_ = iou.result().numpy()
iou.reset_state()

#Calculate true and false positives
TP = np.sum(np.logical_and(predicted_mask == 1, true_mask == 1))
# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(predicted_mask == 0, true_mask == 0))
# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(predicted_mask == 1, true_mask == 0))
# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(predicted_mask == 0, true_mask == 1))
#What proportion of positive identifications was actually correct?
precision = TP/(TP+FP)
#What proportion of actual positives was identified correctly?
recall = TP/(TP+FN)

#plot confusion matrix
row1=[TP, FP]
row2=[FN, TN]
cm = np.array([row1, row2])
cm_norm = np.array([np.array(row1)/np.sum(row1), np.array(row2)/np.sum(row2)])
fontsize=12
plt.ioff()
fig, ax = plt.subplots(1, 1)
sns.heatmap(cm_norm, annot=False,cmap='Blues', ax=ax)
ax.set_xlabel('True Label', fontsize=fontsize+3)
ax.set_ylabel('Predicted',fontsize=fontsize+3)
ax.set_xticks([0.5, 1.5], ['LFE', 'No-LFE'], fontsize=fontsize)
ax.set_yticks([0.5, 1.5], ['LFE', 'No-LFE'], fontsize=fontsize)
for i in range(2):
    for j in range(2):
        text = ax.text(j+0.5, i+0.5, round(cm_norm[i, j],2), fontsize=fontsize,
                       ha="center", va="center", color="k")
plt.tight_layout()
plt.savefig(figure_fp + f'/confusion_matrix_{data_name}.png')
plt.clf()
