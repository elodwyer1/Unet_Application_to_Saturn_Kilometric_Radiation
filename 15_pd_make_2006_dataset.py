# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:04:16 2023

@author: eliza
"""

import numpy as np
from datetime import datetime
import cv2
import matplotlib
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import readsav
import shapely.geometry as sg
import time as t
from tfcat import TFCat
from os import path
import shutil
import os
import sys 
import keras
from os import listdir
from os.path import isfile, join
from matplotlib import colors
import calendar
import configparser

config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'


def return_trajectory_chunk(start, end):
    year= datetime.strftime(start, '%Y')
    traj_df =pd.read_csv(input_data_fp + '/trajectory{}.csv'.format(year), parse_dates=['datetime_ut']) 
    traj_df_chunk=traj_df.loc[traj_df['datetime_ut'].between(start, end),:].reset_index(drop=True)
    if len(traj_df_chunk) == 0:
        traj_df_chunk = traj_df.copy()
    return traj_df_chunk
def interpolate_traj(start, end):
    traj_df_chunk=return_trajectory_chunk(start, end)
        
    lats = traj_df_chunk['lat']
    
    #Normalize to between 0 and 1.
    nrlz_lats = (lats+80)/160
    
    #Interpolate Latitude to the new time values.
    lts = traj_df_chunk['localtime']
    
    #Normalize to between 0 and 1.
    nrlz_lts=lts/24
    
    return nrlz_lats, nrlz_lts
def median_absolute_deviation(data):
    # Calculates the median absolution deviation of the sample
    # Returns the MAD param, and the median
    
    median=np.nanmedian(data)
    
    difs=np.empty(data.size)
    difs[:]=np.nan
    for i in range(data.size):
        difs[i]=abs(data[i]-median)

    mad=np.nanmedian(difs)

    return mad,median
def take_median_std(start, end):
    nrlz_lats, nrlz_lts = interpolate_traj(start, end)
    lt_std, lt_med = median_absolute_deviation(nrlz_lts)
    lat_std, lat_med = median_absolute_deviation(nrlz_lats)
    return lt_med, lt_std, lat_med, lat_std


def get_polygons(polygon_fp,start, end):
    unix_start=t.mktime(start.utctimetuple())
    unix_end=t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array=[]
    if path.exists(polygon_fp):
        catalogue = TFCat.from_file(polygon_fp)
        for i in range(len(catalogue)):
            #if catalogue._data.features[i].properties['feature_type'] !='LFE_sm':
            time_points=np.array(catalogue._data.features[i]['geometry']['coordinates'][0])[:,0]
            if any(time_points <= unix_end) and any(time_points >= unix_start):
                polygon_array.append(catalogue._data.features[i]['geometry']['coordinates'][0])
#polgyon array contains a list of the co-ordinates for each polygon within the time interval           
    return polygon_array


def find_mask(time_view_start, time_view_end, val, polygon_fp,ind,fp_sav):
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
        fund_points = fund_polygon.intersection(data_points)
        if len(fund_points.bounds)>0:
            mask[[int(geom.z) for geom in fund_points.geoms]] = 1
    mask = (mask == 0)
    
    #Set non-polygon values to zero in the signal array.
    flux_ones = np.where(s>0, 1, np.nan)
    v = np.ma.masked_array(flux_ones, mask=mask).filled(np.nan)
    
    #width and height of array (f,t)
    w = len(time_dt64) 
    h = len(frequency)
    

    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)  
    ax.set_axis_off()
    cmap = mpl.colors.ListedColormap(['white']).copy()
    cmap.set_bad('black')
    ax.pcolormesh(time_dt64,frequency, v,cmap=cmap,shading='auto')
    ax.set_axis_off()
    ax.set_yscale('log')
    figure_label = fp_sav + '/mask_images/mask_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    return None

def extract_data(start_ind, end_ind, val):
    # read the save file and copy variables
    
    # copy the flux and frequency variable into temporary variable in
    # order to interpolate them in log scale
    if val == 's':
        s = flux.copy()[:,start_ind:end_ind]
    elif val == 'v':
        s = pol.copy()[:, start_ind:end_ind]

    frequency_tmp = np.load(output_data_fp + '/frequency.npy', allow_pickle=True).copy()
        
    # frequency_tmp is in log scale from f[0]=3.9548001 to f[24] = 349.6542
    # and then in linear scale above so it's needed to transfrom the frequency
    # table in a full log table and einterpolate the flux table (s --> flux
    step = (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/399
    frequency = 10**(np.arange(np.log10(frequency_tmp[0]), np.log10(frequency_tmp[-1])+step/2, step, dtype=float))
    flux_ = np.zeros((frequency.size, s.shape[1]), dtype=float)
    for i in range(s.shape[1]):
        if s.shape[1]!=128:
            break
        flux_[:, i] = np.interp(frequency, frequency_tmp, s[:, i])
    return frequency, flux_


def plot_pol_and_flux(time_view_start, time_view_end, ind,fp_sav):
    
    freq_, pol_ = extract_data(time_view_start, time_view_end,val='v')
    freq_, flux_ = extract_data(time_view_start, time_view_end,val='s')

    time_ = np.arange(pol_.shape[1])
    #Polarization
    vmin =-1
    vmax = 1
    clrmap = 'binary_r'
    scaleZ = colors.Normalize(vmin=vmin, vmax=vmax)
    w = len(time_) 
    h = len(freq_)
    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.pcolormesh(time_, freq_, pol_, norm=scaleZ,cmap=clrmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = fp_sav + '/pol_images/pl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    plt.clf()
      
    
    #Flux
    vmin = 1e-25
    vmax = 1e-19
    scaleZ = colors.LogNorm(vmin, vmax)
    cmap = mpl.cm.get_cmap("binary_r").copy()
    cmap.set_bad('black')
    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.pcolormesh(time_, freq_, flux_, norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_axis_off()
    ax.set_yscale('log')
    figure_label = fp_sav + '/flux_images/fl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    
    return None

def img_to_arr(infilename) :
    img=plt.imread(infilename)
    
    imgr=img[:,:,0]
    imgg=img[:,:,1]
    imgb=img[:,:,2]
    imgalpha=img[:,:,3]

    img_inds=(imgalpha==0)
    imgr=np.ma.masked_array(imgr, mask=img_inds).filled(0).reshape((imgr.shape[0], imgr.shape[1],1))
    imgg=np.ma.masked_array(imgg, mask=img_inds,fill_value=0).filled(0).reshape((imgr.shape[0], imgr.shape[1],1))
    imgb=np.ma.masked_array(imgb, mask=img_inds,fill_value=0).filled(0).reshape((imgr.shape[0], imgr.shape[1],1))
    imgcom=np.concatenate([imgr, imgg, imgb],axis=2)
    img_gray =cv2.cvtColor(imgcom,cv2.COLOR_RGB2GRAY)
    return img_gray

#Function to combine flux and polarization data.
def combine_channels(flux_file, pol_file):
    flx = img_to_arr(flux_file)
    flx=np.reshape(flx, (flx.shape[0], flx.shape[1], 1))
    pol=img_to_arr(pol_file)
    pol=np.reshape(pol, (pol.shape[0], pol.shape[1], 1))
    ttl = np.concatenate([flx, pol], axis=2)
    if ttl.shape[1]!=128:
        sys.exit()
    return ttl 
def read_mask(file1):
    img=plt.imread(file1)
    img_gray =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return img_gray

def save_files(flux_file, pol_file,num,path_):
    ttl = combine_channels(flux_file, pol_file)
    
    ttlpath = path_+str(num).zfill(4)+'/images'
    if os.path.exists(ttlpath):
        shutil.rmtree(ttlpath)
    os.makedirs(ttlpath)
   
    np.save(ttlpath + '/' + str(num).zfill(4)+'.npy',ttl)
    
    return None

def save_label(label,num,path_):
    ttlpath=path_+str(num).zfill(4)+'/label'
    if os.path.exists(ttlpath):
        shutil.rmtree(ttlpath)
    os.makedirs(ttlpath)
    label_path = ttlpath + '/' + str(num).zfill(4)+'.npy'
    np.save(label_path,label)
    return None
def save_traj(lat_st, lat_m, lt_std, lt_med,num,path_):
    ttlpath=path_+str(num).zfill(4)+'/traj'
    
    if os.path.exists(ttlpath):

      shutil.rmtree(ttlpath)
    os.makedirs(ttlpath)

    label_arr = [lat_st, lat_m, lt_std, lt_med]
    
    np.save(ttlpath + '/' + str(num).zfill(4)+'.npy',label_arr)
    return None

'''Make Catalogue of start/stop times and traj. info.'''
#Load start and end times of LFEs and non-LFE
data_start = '2006-01-01T00:00:00'
data_end = '2007-01-01'
data_label1 = datetime.strftime(pd.Timestamp(data_start), '%Y%j')
data_label2 = datetime.strftime(pd.Timestamp(data_end), '%Y%j')
path_to_data =output_data_fp + f"/data_{data_label1}_{data_label2}"
if not os.path.exists(path_to_data):
    os.makedirs(path_to_data)



#make list of start and end times of each model input.
t_start = np.arange(np.datetime64(data_start), np.datetime64(data_end), np.timedelta64(219, "m"))
t_end = np.array([i+np.timedelta64(384, "m") for i in t_start])
t_timestamp = np.load(input_data_fp + "/time_indatetime_all_years.npy", allow_pickle=True)
t_timestamp = t_timestamp[(t_timestamp >= t_start[0]) & (t_timestamp <= t_end[-1])]
t_isostring = np.array([datetime.strftime(i,'%Y-%m-%dT%H:%M:%S') for i in t_timestamp])
time = np.array(t_isostring, dtype=np.datetime64)
div = len(t_timestamp)%73
inds_split_s = np.arange(0, len(t_timestamp)-128, 73)
inds_split_e = [i + 128 for i in inds_split_s[:-1]]
inds_split_e.append(len(t_timestamp)-1)
t_start = t_timestamp[inds_split_s]
t_end = t_timestamp[inds_split_e]
total_df=pd.DataFrame({'start': t_start, 'end':t_end})


#Find corresponding trajectory data for trajectory color channels of input.
vals = np.array(list(map(take_median_std,total_df['start'] , total_df['end'])))
total_df['lt_median']=vals[:,0]
total_df['lt_stdev']=vals[:,1]
total_df['lat_median']=vals[:,2]
total_df['lat_stdev']=vals[:,3]
total_df.to_csv(path_to_data + "/catalogue.csv",index=False)

''' Save flux and polarization images'''
#Make filepaths for images to be saved.
flux_fp = path_to_data + '/flux_images'
if not os.path.exists(flux_fp):
    os.makedirs(flux_fp)
pol_fp = path_to_data + '/pol_images'
if not os.path.exists(pol_fp):
    os.makedirs(pol_fp)

flux = np.load(input_data_fp + "/s_all_years.npy", allow_pickle=True)
pol = np.load(input_data_fp + "/p_all_years.npy", allow_pickle=True)
total_df=pd.read_csv(path_to_data + "/catalogue.csv", parse_dates=['start','end'])
_index = np.arange(len(total_df))

for day1, day2, i in zip(inds_split_s, inds_split_e, _index):
    plt.ioff()
    fp_sav = path_to_data
    #make flux and polarization spectrograms and save as images.
    plot_pol_and_flux(day1, day2, i, fp_sav)

''' Save images to format for Model '''

total_df=pd.read_csv(path_to_data + "/catalogue.csv", parse_dates=['start','end'])
#number of images
count = len(total_df)
#Read in directories for each Flux/Polarization Spectrogram showing an LFE image.

flxpath= path_to_data + "/flux_images/"
flx_labels=[f for f in listdir(flxpath) if isfile(join(flxpath, f))]
flxfiles = [flxpath+f for f in flx_labels]
polpath=path_to_data + "/pol_images/"
pol_labels=[f for f in listdir(polpath) if isfile(join(polpath, f))]
polfiles = [polpath+f for f in pol_labels]


#LFE filepaths
flux_lfe_files = [flxpath+'fl_img'+str(i).zfill(4)+'.png' for i in range(count)]
#Final List
flux_lfe_files = [i for i in flux_lfe_files if i in flxfiles]
###############################################################
pol_lfe_files=[polpath+'pl_img'+str(i).zfill(4)+'.png' for i in range(count)]
#Final List
pol_lfe_files=[i for i in pol_lfe_files if i in polfiles]


lat_s=total_df['lat_stdev']
lat_m=total_df['lat_median']
lt_s=total_df['lt_stdev']
lt_m=total_df['lt_median']


path_ =  path_to_data + f'/test_{data_label1}_{data_label2}/'
[save_traj(lat_s[i], lat_m[i], lt_s[i], lt_m[i],i,path_) for i in range(len(lat_s))]
[save_files(flux_lfe_files[i], pol_lfe_files[i], i,path_) for i in range(len(flux_lfe_files))]
