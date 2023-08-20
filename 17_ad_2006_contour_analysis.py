# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:30:14 2023

@author: eliza
"""

import json
from datetime import datetime
from shapely.validation import make_valid
import cv2
import itertools
import numpy as np
import pandas as pd
from astropy.time import Time
from tfcat import TFCat
import matplotlib.pyplot as plt
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
def get_contours(mask):
    binary=np.where(mask>0.5, 1, 0)
    cv2img = binary.astype(np.uint8)
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
        mask = results[:, :].copy()
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
    fmapped_max = [max(i[:,1]) for i in contours_total_]
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
    
    del_f_khz = [(j-i) for i, j in zip(fmapped_min, fmapped_max)]  
    
    df = pd.DataFrame({'area':area,'min_f':fmapped_min, 'delta_f':del_f,
                       'delta_t':del_t, 'id':im_id, 'delta_f_khz':del_f_khz})
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
    for i in range(len(catalogue)):
        label=catalogue._data.features[i]['properties']['feature_type']
        feature.append(label)
        id_.append(catalogue._data.features[i]['id'])
        coords=np.array(catalogue._data.features[i]['geometry']['coordinates'][0])
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
def calculate_probability(res_, contour):
    res2 = res_[:, min(contour[:, 0]):max(contour[:, 0]+1)]
    t=np.arange(min(contour[:, 0]), max(contour[:, 0]+1), 1)
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
        t_inds = [t_proxy.index(i) for i in contour[:,0]]
        mask[contour[:,1], t_inds] = 1
        
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
def twodhist(df):
    scatter_x = np.array(df['delta_f'])
    scatter_y = np.array(df['delta_t'])
    xedges=np.arange(0, 390, 5)
    yedges = np.arange(0, 135, 5)

    H, x, y = np.histogram2d(scatter_x, scatter_y, bins=(xedges, yedges))
    H=(H).T
    return H
def make_dataframe(file):
    #Timestamps is in the form of pandas timestamp, but you can edit the lfe_coordinates function
    #if you would like it in a different format.
    timestamps, freqs, feature, id_ = lfe_coordinates(file)
    
    
    #Start and end times of each labelled item.
    start = [min(i) for i in timestamps]
    end = [max(i) for i in timestamps]
    
    df = pd.DataFrame({'start': start, 'end':end,'label':feature,'id':id_})
    df = df.drop_duplicates(subset='start', keep='first')
    df=df.sort_values(by='start').reset_index(drop=True)
    return df
def write_to_json(fp_save, t_unix_utc, f, labs):
    with open(fp_save, 'w') as js_file:
        TFCat = {"type": "FeatureCollection", "features": [], "crs": {"type": "local", "properties": { "name": "Time-Frequency", "time_coords_id": "unix","spectral_coords": {"type": "frequency", "unit": 'kHz'}, "ref_position_id": 'Cassini'}}}
        count = 0
        for i, j,k in zip(t_unix_utc, f,labs):
            coords = [[[x,y] for x,y in zip(i,j)]]
            TFCat['features'].append({"type": "Feature", "id": count, "geometry": {"type": "Polygon", "coordinates": coords}, "properties": {"feature_type": k}})
            count += 1
        json.dump(TFCat, js_file)
    return None
def make_hist_norm(ax, df, label, bins,percentiles,fontsize):
    hist_, bins_ = np.histogram(df[label], bins)
    norm_hist_ = hist_/len(df)
    ax.tick_params(labelsize=12)
    ax.hist(bins_[:-1], bins_, weights=norm_hist_, color='steelblue')
    ax.set_ylabel('Normalised Counts', fontsize=fontsize)
    
    ax.set_ylim(0, max(norm_hist_)+0.1)

    if percentiles == True:
        median_ = round(np.percentile(df[label], 50), 2)
        p75 = round(np.percentile(df[label], 75), 2)
        p25 = round(np.percentile(df[label], 25), 2)
        ax.vlines(p75, 0, max(norm_hist_)+0.1, linestyle='dotted', label='75th percentile={}'.format(p75))
        ax.vlines(median_, 0, max(norm_hist_)+0.1, linestyle='dotted',label=f'50th percentile={median_}')
        ax.vlines(p25, 0, max(norm_hist_)+0.1, linestyle='dotted', label='25th percentile={}'.format(p25))
        ax.legend(fontsize=fontsize-4)
    return ax

thresh = np.load(output_data_fp + f'/{model_name}/best_thresh.npy')
data_name = 'test_2006001_2007001'
###### Analyse Test Set ###############################

test_path = output_data_fp + f"/{data_name}"
test_res = np.load(output_data_fp + f'/{model_name}/{data_name}_av_overlap_combined.npy', \
                  allow_pickle=True)
    
test_results=np.where(test_res.copy()>=thresh, 1, 0)

#find all contours in array of 2006
test_im_id, test_contours_total_, test_contours_unmapped= total_contours([0], test_results)
test_df = analyse_contours([0],test_results)
# Separate polygons according to criterion
test_selected = test_df.loc[(test_df['delta_f']>=100) & (test_df['min_f'] < 100), :]
test_selected_inds = np.array(test_selected.index).astype(int)
selected_contours = [test_contours_total_[i] for i in test_selected_inds]
'''
#Find indices of unselected contours.
test_unselected_inds = [i for i in range(len(test_df)) if i not in test_selected_inds]
#Make dataframe with features describing each contour for unselected data
test_unselected = test_df.loc[test_unselected_inds, :].reset_index(drop=True)
#calculate average probability of selected contours
probability = []
for i in test_selected_inds:
    print(i)
    p = calculate_probability(test_res, test_contours_unmapped[i])
    probability.append(p)
test_selected['probability'] = probability
test_selected.to_csv(output_data_fp + f'/{model_name}/{data_name}_selected_contour_analysis.csv',
                     index=False)
'''
############### Plot contour properties ###################
'''
test_selected = pd.read_csv(output_data_fp + f'/{model_name}/{data_name}_selected_contour_analysis.csv')
#plot of delta t vs delta v with points coloured by output probability.
plt.tick_params(labelsize=15)
plt.ioff()
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_title('LFEs detected in 2006 using input method with overlapping time ranges', 
             fontsize=12)

t_hours = (test_selected['delta_t'].reset_index(drop=True) * 3)/60
im = ax.scatter(t_hours, test_selected['delta_f_khz'], s=8,
            c=test_selected['probability'], cmap='Blues')
ax.set_xscale('log')
ax.set_xlabel('Delta t (hours)', fontsize=12)
ax.set_ylabel('Delta f (kHz)', fontsize=12)
cbar = fig.colorbar(im, label='Probability')
plt.savefig(output_data_fp + f'/{model_name}/figures/{data_name}_probability_dt_df.png')
'''



############## Convert to actual frequency and time values to extract polygon coords ##
#times in unix that correspond to the pixel indices
t_dt = np.load(input_data_fp + "/time_indatetime_all_years.npy", allow_pickle=True)
df_2006 = pd.read_csv(output_data_fp + "/data_2006001_2007001/catalogue.csv",
                      parse_dates=['start','end'])
time_view = t_dt.copy()[(t_dt >= df_2006['start'].iloc[0]) & (t_dt <= df_2006['end'].iloc[-1])]
t_isostring = np.array([datetime.strftime(i,'%Y-%m-%dT%H:%M:%S') for i in time_view])
t_dt64 = np.array(t_isostring, dtype=np.datetime64)
time_unix=np.array([i.astype('uint64').astype('uint32') for i in t_dt64])

#Convert from pixel number to unix value.
time_contours_unix =[]
f_contours_khz = []
for i in selected_contours:
    timez = i[:,0].astype(int)
    timez_unix = time_unix[timez].astype(float)
    time_contours_unix.append(timez_unix)
    f_contours_khz.append( i[:,1])
#save to .json file
fp_sav = output_data_fp + f'/{model_name}/{data_name}_selected_contours.json'
lab = np.repeat('LFE', len(time_contours_unix))
write_to_json(fp_sav, time_contours_unix, f_contours_khz, lab)
df = make_dataframe(fp_sav)
df.to_csv(output_data_fp + f'/{model_name}/{data_name}_catalogue.csv',index=False)
