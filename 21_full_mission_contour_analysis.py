# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:15:20 2023

@author: eliza
"""
import json
from datetime import datetime
from shapely.validation import make_valid
import cv2
from shapely.geometry import LinearRing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from tfcat import TFCat
from IPython import get_ipython
from shapely.geometry import Point, Polygon, LineString, MultiPoint
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']


def plot_spectrogram(ax, flux):
    
    frequency = np.flip(10**(np.arange(np.log10(3.95), np.log10(1500), \
                (np.log10(1500)-np.log10(3.95))/384, dtype=float)))
    time = np.arange(flux.shape[1])
    fontsize=14
    
    im = ax.pcolormesh(time, frequency, flux, cmap='binary_r')
    ax.set_yscale('log')
    ax.tick_params(labelsize=fontsize)


    return im    
def get_contours(result, thresh):    
    binary=np.where(result.copy()>=thresh, 1, 0)
    cv2img = binary.astype(np.uint8)
    contours, hierarchy = cv2.findContours(cv2img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    reshape_contours = lambda x: np.reshape(x, (x.shape[0], x.shape[-1]))
    reshaped_contours = [reshape_contours(i) for i in contours]
    return reshaped_contours
def map_frequencies(contour_):
    frequency_tmp = np.load(output_data_fp + '/frequency.npy', allow_pickle=True).copy()
    step = (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/383
    frequency = np.flip(10**(np.arange(np.log10(frequency_tmp[0]),
                                       np.log10(frequency_tmp[-1])+step/2, step, 
                                       dtype=float)))
    y = np.array(contour_[:,1])
    y_mapped = frequency[y]
    x=contour_[:,0]
    mapped_coords = np.array([(i,j) for i,j in zip(x, y_mapped)])
    return mapped_coords
def total_contours(results, thresh):
    
    mask = results[:, :].copy()
    contours_unmapped = get_contours(mask, thresh)
    mapped_contours = [map_frequencies(i) for i in contours_unmapped]

    return mapped_contours, contours_unmapped
def analyse_contours(results, thresh):
    contours_total_, contours_unmapped = total_contours(results, thresh)
    ### Delta F
    fmapped_min = [min(i[:,1]) for i in contours_total_]
    fmapped_max = [max(i[:,1]) for i in contours_total_]
    min_f = []
    max_f = []
    area = []
    min_t = []
    max_t = []
    count=0
    for i in contours_unmapped:
        print(f'Find properties of contours {count+1} of {len(contours_unmapped)}')
        area.append(cv2.contourArea(i))
        min_f.append(min(i[:,1]))
        max_f.append(max(i[:,1]))
        min_t.append(min(i[:,0]))
        max_t.append(max(i[:,0]))
        count+=1
    del_f = [(j-i) for i, j in zip(min_f, max_f)]   
    del_t = [(j-i) for i, j in zip(min_t, max_t)]  
    
    del_f_khz = [(j-i) for i, j in zip(fmapped_min, fmapped_max)]  
    
    df = pd.DataFrame({'area':area,'min_f':fmapped_min, 'delta_f':del_f,
                       'delta_t':del_t, 'delta_f_khz':del_f_khz})
    return df
### remove misclassified polygons from masks.
def remove(un_df, results, c_unmapped):
    mask = results[:, :].copy()
    contour_inds = un_df.reset_index(drop=True)
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
def get_data(file):
    co = []
    id_ = []
    feature=[]
    catalogue = TFCat.from_file(file)
    for i in range(len(catalogue._data['features'])):
        label=catalogue._data['features'][i]['properties']['feature_type']
        feature.append(label)
        id_.append(catalogue._data['features'][i]['id'])
        coords=np.array(catalogue._data['features'][i]['geometry']['coordinates'][0])
        co.append(coords)
        
    return co, id_, feature

def lfe_coordinates(file):
    co, id_, feature = get_data(file)
    timestamps = []
    freqs = []
    for i in range(len(co)):
        time_points=Time(co[i][:,0],format='unix').to_value('isot')
        f_points=co[i][:,1]
        timestamps.append([datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in time_points])
        freqs.append(f_points)
    return timestamps, freqs, feature, id_

def make_dataframe(timestamps, freqs):
    #Timestamps is in the form of pandas timestamp, but you can edit the lfe_coordinates function
    #if you would like it in a different format.
    #Start and end times of each labelled item.
    start = [min(i) for i in timestamps]
    end = [max(i) for i in timestamps]
    start_iso =Time(start,format='unix').to_value('isot')
    end_iso = Time(end,format='unix').to_value('isot')
    start_dt = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in start_iso]
    end_dt = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in end_iso]
    df = pd.DataFrame({'start': start_dt, 'end':end_dt})
    return df
def write_json(fp_save, t_unix_utc, f, labs):
    with open(fp_save, 'w') as js_file:
        TFCat = {"type": "FeatureCollection", "features": [],
                 "crs": {"name": "Time-Frequency",
                         "properties": {"time_coords": {"id": "unix", "name": "Timestamp (Unix Time)", "unit": "s", "time_origin": "1970-01-01T00:00:00.000Z", "time_scale": "TT"},
                                        "spectral_coords": {"name": "Frequency", "unit": "kHz"}, 
                                        "ref_position": {"id": "Cassini"}}}}
        count = 0
        for i, j,k in zip(t_unix_utc, f,labs):
            coords = [[[x,y] for x,y in zip(i,j)]]
            # polygon coordinates need to be in counter-clockwise order (TFCat specification)
            if (LinearRing(coords[0])).is_ccw == False:
                coords = [coords[0][::-1]]
            TFCat['features'].append({"type": "Feature", "id": count, "geometry": {"type": "Polygon", "coordinates": coords}, "properties": {"feature_type": k}})
            count += 1

        json.dump(TFCat, js_file)
def calculate_probability(res_, contour):
    res2 = res_[:, min(contour[:, 0]):max(contour[:, 0])]
    t=np.arange(min(contour[:, 0]), max(contour[:, 0]), 1)
    f = np.arange(res2.shape[0])
    times, freqs=np.meshgrid(t, f)
    #Total length of 2D signal array.
    data_len = len(res_.flatten())
    #indices of each item in flattened 2D signal array.
    index = np.arange(data_len, dtype=int)
    #Co-ordinates of each item in 2D signal array.
    coords = [(t, f) for t,f in zip(times.flatten(), freqs.flatten())]
    data_points = MultiPoint([Point(x, y, z) for (x, y), z in zip(coords, index)])
    #Make mask array.
    mask = np.zeros(res2.shape)
    t_proxy = list(t)
    #Find overlap between polygons and signal array.
    #Set points of overlap to 1.
    
    if (contour.shape[0]==1):
        mask[contour[:,1], contour[:,0]] = 1
        
    elif (contour.shape[0]<3) and (contour.shape[0]>1):
        line_coords = np.array([(i, j) for i,j in zip(contour[:, 0], contour[:, 1])])
        line = LineString(line_coords)
        fund_points = line.intersection(data_points)
        fund_coords = np.array([i.coords for i in fund_points.geoms]).astype(int)[:,0,0:2]
        t_inds = [np.where(t_proxy==i)[0] for i in fund_coords[:, 0]]
        mask[fund_coords[:, 1], t_inds] = 1
        
    else:
        fund_polygon = Polygon(contour)
        valid_shape = make_valid(fund_polygon)
        fund_points = valid_shape.intersection(data_points)
        fund_coords = np.array([i.coords for i in fund_points.geoms]).astype(int)[:,0,0:2]
        t_inds = [t_proxy.index(i) for i in fund_coords[:, 0]]
        mask[list(fund_coords[:, 1]), t_inds] = 1
    ind1, ind2 = np.nonzero(mask)
    v = res2.copy()[ind1, ind2]
    prob = np.mean(v)
    return prob
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

#load probability value for thresholding prediction.
thresh = np.load(output_data_fp + f'/{model_name}/best_thresh.npy')
data_name = '2004001_2017258'
###### Analyse Test Set ###############################
test_path = output_data_fp + f"/{data_name}"
#load 2d array with results for entire dataset
#The array was padded with zeros so that inputs are 128 pixels wide exactly
#for that reason we clip the results to only contain the real data.
test_res = np.load(output_data_fp + f'/{model_name}/{data_name}_av_overlap_combined.npy', \
                 allow_pickle=True)[:, :2403360]
#thresold to convery to binary mask
test_results=np.where(test_res.copy()>=thresh, 1, 0)
#Find contours from mask
test_contours_total_, test_contours_unmapped= total_contours(test_res, thresh)
#dataframe summarising properties of contours 
#contains columns 'area', 'min_f'(in kHz),  'delta_f', 'delta_t','delta_f_khz'
test_df = analyse_contours(test_res, thresh)
# Separate polygons according to criterion
test_selected = test_df.loc[(test_df['delta_f']>=100) & (test_df['min_f'] < 100), :]
'''
#Calculate the average probability within the selected polygons
probability = []
count=0
for i in test_selected.index:
    print(count/len(test_selected.index))
    p = calculate_probability(test_res, test_contours_unmapped[i])
    probability.append(p)
    count+=1
test_selected['probability'] = np.array(probability)
#save dataframe to output data folder.
test_selected.to_csv(output_data_fp + f'/{model_name}/{data_name}_selected_contours.csv',
                     index=False)
'''

##### Convert from binary Mask to actual frequency-time coords ###
#times in unix that correspond to the pixel indices

t_dt = np.load(input_data_fp + "/time_indatetime_all_years.npy", allow_pickle=True)
t_isostring = np.array([datetime.strftime(i,'%Y-%m-%dT%H:%M:%S') for i in t_dt])
t_dt64 = np.array(t_isostring, dtype=np.datetime64)
#all time values in cassini radio data
time_unix=np.array([i.astype('uint64').astype('uint32') for i in t_dt64])


#Contours that satisfy criterion to be converted to frequency-time coords
test_selected_inds = np.array(test_selected.index).astype(int)
selected_contours = [test_contours_total_[i] for i in test_selected_inds]
#empty lists to be filled
time_contours_unix =[]
f_contours_khz = []
count=0
for i in test_selected_inds:
    print(f'Find freq-time coords of {count+1} of {len(test_selected_inds)} LFEs')
    timez = test_contours_unmapped[i][:,0].astype(int)
    timez_unix = time_unix[timez].astype(float)
    time_contours_unix.append(timez_unix)
    f_contours_khz.append(test_contours_total_[i][:,1])
    count+=1
    

df = make_dataframe(time_contours_unix, f_contours_khz)
dupes = df.loc[df.duplicated(subset=['start'], keep=False), :]
df_ = df.sort_values(by='end')
df_ = df_.drop_duplicates(subset='start', keep='last')
inds_keep = np.array(df_.index)
t_final = np.array(time_contours_unix,dtype=object)[inds_keep]
f_final = np.array(f_contours_khz,dtype=object)[inds_keep]

fp_sav = output_data_fp + f"/{model_name}/{data_name}_catalogue.json"
#write_json(fp_sav, time_contours_unix, f_contours_kHz)
#df_.to_csv(output_data_fp + f"/{model_name}/{data_name}_catalogue.csv", index=False)
