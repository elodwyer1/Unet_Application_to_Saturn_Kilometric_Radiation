# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:19:01 2023

@author: eliza
"""
## Imports
import shutil
import os
import random
import numpy as np
from matplotlib import colors
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.io import readsav
from sklearn.model_selection import train_test_split
from mpl_toolkits import axes_grid1
from os import listdir
from os.path import isfile, join
from datetime import datetime
import tensorflow as tf
from keras import layers
import time as t
from tfcat import TFCat
import shapely.geometry as sg
from astropy.time import Time
from os import path
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
## Seeding 
seed = 42
random.seed = seed
np.random.seed = seed

def get_polygons(polygon_fp,start, end):
    unix_start=t.mktime(start.utctimetuple())
    unix_end=t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array=[]
    if path.exists(polygon_fp):
        catalogue = TFCat.from_file(polygon_fp)
        for i in range(len(catalogue)):
                if catalogue._data.features[i].properties['feature_type'] =='LFE_sm':
                    time_points=np.array(catalogue._data.features[i]['geometry']['coordinates'][0])[:,0]
                    if any(time_points <= unix_end) and any(time_points >= unix_start):
                        polygon_array.append(catalogue._data.features[i]['geometry']['coordinates'][0])
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval           
    return polygon_array


def find_mask(time_view_start, time_view_end, val, file_data,polygon_fp,ind,fp_sav):
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval
    polygon_array=get_polygons(polygon_fp, time_view_start, time_view_end)
    #signal data and time frequency values within the time range specified.
    time_dt64, frequency, flux=extract_data(file_data, time_view_start, time_view_end, val)
    time_unix=[i.astype('uint64').astype('uint32') for i in time_dt64]
    #Meshgrid of time/frequency vals.
    times, freqs=np.meshgrid(time_unix, frequency)
    #Total length of 2D signal array.
    data_len = len(flux.flatten())
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
    flux_ones = np.where(flux>0, 1, np.nan)
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

def extract_data(file_data, time_view_start, time_view_end, val):
    # read the save file and copy variables
    time_index = 't'
    freq_index = 'f'
    val_index = val
    file = readsav(file_data)
    t_doy = file[time_index].copy()
    doy_one = pd.Timestamp(str(1997)) - pd.Timedelta(1, 'D')
    t_timestamp = np.array([doy_one + pd.Timedelta(t * 1440, 'm') for t in t_doy],
        dtype=pd.Timestamp)
    t_isostring = np.array([datetime.strftime(i,'%Y-%m-%dT%H:%M:%S') for i in t_timestamp])
    time =t_isostring
    time = np.array(time, dtype=np.datetime64)
    time_view = time[(time >= time_view_start) & (time < time_view_end)]
    # copy the flux and frequency variable into temporary variable in
    # order to interpolate them in log scale
    s = file[val_index][:, (time >= time_view_start) & (time < time_view_end)].copy()
    frequency_tmp = file[freq_index].copy()
    # frequency_tmp is in log scale from f[0]=3.9548001 to f[24] = 349.6542
    # and then in linear scale above so it's needed to transfrom the frequency
    # table in a full log table and einterpolate the flux table (s --> flux
    step = (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/383
    frequency = 10**(np.arange(np.log10(frequency_tmp[0]), np.log10(frequency_tmp[-1])+step/2, step, dtype=float))
    flux = np.zeros((frequency.size, len(time_view)), dtype=float)
    for i in range(len(time_view)):
        flux[:, i] = np.interp(frequency, frequency_tmp, s[:, i])
    return time_view, frequency, flux


def plot_pol_and_flux(time_view_start, time_view_end, file,ind,fp_sav):
    time, freq, pol = extract_data(
        file, time_view_start=time_view_start, time_view_end=time_view_end,val='v')
    time, freq, flux = extract_data(
        file, time_view_start=time_view_start, time_view_end=time_view_end,val='s')
    #Polarization
    vmin =-1
    vmax = 1
    clrmap = 'binary_r'
    scaleZ = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    w = len(time) 
    h = len(freq)
    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.pcolormesh(time, freq, pol, norm=scaleZ,cmap=clrmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = fp_sav + '/pol_images/pl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    plt.clf()
    #Flux
    #vmin, vmax = flux_norm(time_view_start, time_view_end)
    vmin = 1e-25
    vmax = 1e-19
    scaleZ = mpl.colors.LogNorm(vmin, vmax)
    cmap = mpl.cm.get_cmap("binary_r").copy()
    cmap.set_bad('black')
    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.pcolormesh(time, freq, flux, norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_axis_off()
    ax.set_yscale('log')
    figure_label = fp_sav + '/flux_images/fl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    
    return None

def load_image(id_name, path):
    
    #path = root + 'ML_Dataset/Main_Dataset/train'
    image_h = 384
    image_w = 128
    
    image_path = os.path.join(path, id_name, "images", id_name) + ".npy"
    
    ## Reading Image
    _image = np.load(image_path, allow_pickle=True)
    resize = tf.keras.Sequential([layers.Resizing(image_h, image_w)])
    image = resize(_image)
    
    return image
def load_mask(id_name, path):
    #path = root + 'ML_Dataset/Main_Dataset/train'
    image_h = 384
    image_w = 128
    
    #image_path = os.path.join(path, id_name, "images", id_name) + ".npy"
    mask_path = os.path.join(path, id_name, "masks", id_name) + ".npy"
    
    ## Reading Image
    #image = np.load(image_path, allow_pickle=True)
    #resize = tf.keras.Sequential([layers.Resizing(image_h, image_w)])
    #image = resize(image)
    
    # Reading mask
    _mask_image = np.load(mask_path, allow_pickle=True)
    _mask_image=np.reshape(_mask_image,(_mask_image.shape[0],_mask_image.shape[1],1))
    resize = tf.keras.Sequential([layers.Resizing(image_h, image_w),layers.Rescaling(1./255)])
    mask = resize(_mask_image)[:,:,0]
    mask=np.where(mask>0, 1, 0).reshape(image_h, image_w,1)
    return mask

#Function to read image and convert to numpy array.
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
    return ttl 
def read_mask(file1):
    img=plt.imread(file1)
    img_gray =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return img_gray

def save_files(flux_file, pol_file,mask_file1,num,path_):
    ttl = combine_channels(flux_file, pol_file)
    mask = read_mask(mask_file1)
    
    ttlpath = path_+str(num).zfill(3)+'/images'
    if os.path.exists(ttlpath):
        shutil.rmtree(ttlpath)
    os.makedirs(ttlpath)
    maskpath = path_+str(num).zfill(3)+'/masks'
    if os.path.exists(maskpath):
        shutil.rmtree(maskpath)
    os.makedirs(maskpath)
    np.save(ttlpath + '/' + str(num).zfill(3)+'.npy',ttl)
    np.save(maskpath + '/' + str(num).zfill(3)+'.npy',mask)
    return ttl,mask

def save_label(label,num,path_):
    ttlpath=path_+str(num).zfill(3)+'/label'
    if os.path.exists(ttlpath):
        shutil.rmtree(ttlpath)
    os.makedirs(ttlpath)
    label_path = ttlpath + '/' + str(num).zfill(3)+'.npy'
    np.save(label_path,label)
    return None
def save_traj(lat_st, lat_m, lt_std, lt_med,num,path_):
    ttlpath=path_+str(num).zfill(3)+'/traj'
    
    if os.path.exists(ttlpath):

      shutil.rmtree(ttlpath)
    os.makedirs(ttlpath)

    label_arr = [lat_st, lat_m, lt_std, lt_med]
    
    np.save(ttlpath + '/' + str(num).zfill(3)+'.npy',label_arr)

    return None
def return_trajectory_chunk(start, end):
    year= datetime.strftime(start, '%Y')
    traj_df =pd.read_csv(input_data_fp + '/trajectory{}.csv'.format(year), parse_dates=['datetime_ut']) 
    traj_df_chunk=traj_df.loc[traj_df['datetime_ut'].between(start, end),:].reset_index(drop=True)
    if len(traj_df_chunk) == 0:
        traj_df_chunk = traj_df.copy()
    return traj_df_chunk
def interpolate_traj(start, end):
    #use nearest neighbour interpolation, same as used in the image resizing.

    #print(start)
    traj_df_chunk=return_trajectory_chunk(start, end)
        
    lats = traj_df_chunk['lat_krtp']
    
    
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
    print(start)
    return lt_med, lt_std, lat_med, lat_std


#Load start and end times of LFE_sm
polygon_fp = input_data_fp + '/SKR_LFEs.json'
df=pd.read_csv(input_data_fp + "/total_timestamps.csv", parse_dates=['start','end'])
lfe_df_sm = df.loc[df['label'] == 'LFE_sm'].reset_index(drop=True)
start = lfe_df_sm['start']
end=lfe_df_sm['end']
lfe_index = np.arange(len(lfe_df_sm))
count=0
if not path.exists(output_data_fp + '/lfe_sm/flux_images/'):
    os.makedirs(output_data_fp + '/lfe_sm/flux_images/')
if not path.exists(output_data_fp + '/lfe_sm/pol_images/'):
    os.makedirs(output_data_fp + '/lfe_sm/pol_images/')
if not path.exists(output_data_fp + '/lfe_sm/mask_images/'):
    os.makedirs(output_data_fp + '/lfe_sm/mask_images/')

for day1, day2, i in zip(start, end,lfe_index):
    year = datetime.strftime(day1, '%Y')
    if year == '2017':
        file = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file = input_data_fp + '/SKR_{}_CJ.sav'.format(year)
    plt.ioff()
    fp_sav = output_data_fp + "/lfe_sm"
    #make flux and polarization spectrograms and save as images.
    plot_pol_and_flux(day1, day2, file, i,fp_sav)
    val='s'
    #make corresponding masked spectrogram and save as image.
    a=find_mask(day1, day2, val, file,polygon_fp,i,fp_sav) 
    print(i)

'''________Flux and Pol_________'''

count=1938
#Read in directories for each Flux/Polarization Spectrogram showing an LFE image.
flxpath=output_data_fp + '/lfe_sm/flux_images/'
flx_labels=[f for f in listdir(flxpath) if isfile(join(flxpath, f))]
flxfiles = [output_data_fp + '/lfe_sm/flux_images/'+f for f in flx_labels]
polpath=output_data_fp + '/lfe_sm/pol_images/'
pol_labels=[f for f in listdir(polpath) if isfile(join(polpath, f))]
polfiles = [output_data_fp + '/lfe_sm/pol_images/'+f for f in pol_labels]


#LFE filepaths
flux_lfe_files = [output_data_fp + '/lfe_sm/flux_images/fl_img'+str(i).zfill(4)+'.png' for i in range(count)]
#Final List
flux_lfe_files = [i for i in flux_lfe_files if i in flxfiles]

###############################################################
pol_lfe_files=[output_data_fp + '/lfe_sm/pol_images/pl_img'+str(i).zfill(4)+'.png' for i in range(count)]
#Final List
pol_lfe_files=[i for i in pol_lfe_files if i in polfiles]


'''______Mask_________'''

maskpath1=output_data_fp + '/lfe_sm/mask_images/'
mask_image_label1=[f for f in listdir(maskpath1) if isfile(join(maskpath1, f))]
mask_fp1 = [maskpath1+i for i in mask_image_label1]
mask_files1 = [output_data_fp + '/lfe_sm/mask_images/mask_img'+str(i).zfill(4)+'.png' for i in range(count)]
mask_file1=[i for i in mask_files1 if i in mask_fp1]
print(len(pol_lfe_files), len(flux_lfe_files), len(mask_file1))

'''________labels and trajectory________'''

labels=lfe_df_sm['label']
vals = np.array(list(map(take_median_std, lfe_df_sm['start'] , lfe_df_sm['end'])))
lfe_df_sm['lt_median']=vals[:,0]
lfe_df_sm['lt_stdev']=vals[:,1]
lfe_df_sm['lat_median']=vals[:,2]
lfe_df_sm['lat_stdev']=vals[:,3]
lat_std = np.array(lfe_df_sm['lat_stdev'])
lat_med = np.array(lfe_df_sm['lat_median'])
lt_std = np.array(lfe_df_sm['lt_stdev'])
lt_med = np.array(lfe_df_sm['lt_median'])
lfe_df_sm.to_csv(output_data_fp + '/lfe_sm/lfesm_total_catalogue.csv', index=False)


path_ = output_data_fp + '/lfe_sm/test_lfesm/'
[save_label(labels[i], i,path_) for i in range(len(labels))]
[save_traj(lat_std[i], lat_med[i], lt_std[i], lt_med[i],i,path_) for i in range(len(lat_med))]
[save_files(flux_lfe_files[i], pol_lfe_files[i], mask_file1[i], i,path_) for i in range(len(flux_lfe_files))]


