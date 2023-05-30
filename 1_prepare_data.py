# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:44:33 2022

@author: eliza
"""
import matplotlib
import matplotlib as mpl
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import readsav
import shapely.geometry as sg
import time as t
from tfcat import TFCat
from os import (path, makedirs)
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']


def load_ephem(dtime,dtime2):
    year= pd.to_datetime(dtime).year
    orbit_df = pd.read_csv(input_data_fp +'/trajectory{}.csv'.format(year), parse_dates=['datetime_ut'])
    orbit_df = orbit_df.loc[orbit_df['datetime_ut'].between(dtime,dtime2), :]
    return orbit_df
def get_polygons(polygon_fp,start, end):
    unix_start=t.mktime(start.utctimetuple())
    unix_end=t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array=[]
    if path.exists(polygon_fp):
        catalogue = TFCat.from_file(polygon_fp)
        for i in range(len(catalogue)):
            if catalogue._data.features[i].properties['feature_type'] !='LFE_sm':
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
    
    #Make figure
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

polygon_fp= input_data_fp + "/SKR_LFEs.json"
#Load start and end times of LFEs and non-LFEs
total_df=pd.read_csv(input_data_fp + "/total_timestamps.csv", parse_dates=['start','end'])
total_df_nosm = total_df.loc[total_df['label']!='LFE_sm', :].reset_index(drop=True)
start = total_df_nosm['start']
end=total_df_nosm['end']
lfe_index = np.arange(len(total_df_nosm))
#make path to save flux, polarization and mask images
if not path.exists(output_data_fp + '/flux_images/'):
    makedirs(output_data_fp + '/flux_images/')
if not path.exists(output_data_fp + '/pol_images/'):
    makedirs(output_data_fp + '/pol_images/')
if not path.exists(output_data_fp + '/mask_images/'):
    makedirs(output_data_fp + '/mask_images/')
#make each image
for day1, day2, i in zip(start, end,lfe_index):
    year = datetime.strftime(day1, '%Y')
    if year == '2017':
        file = input_data_fp +'/SKR_2017_001-258_CJ.sav'
    else: 
        file = input_data_fp + '/SKR_{}_CJ.sav'.format(year)
    plt.ioff()
    #make flux and polarization spectrograms and save as images.
    plot_pol_and_flux(day1, day2, file, i,output_data_fp)
    val='s'
    polygon_fp= input_data_fp + "/SKR_LFEs.json"
    #make corresponding masked spectrogram and save as image.
    a=find_mask(day1, day2, val, file, polygon_fp, i, output_data_fp) 
    print(i)
    