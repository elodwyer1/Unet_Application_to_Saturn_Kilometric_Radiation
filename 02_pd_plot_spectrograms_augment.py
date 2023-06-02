# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:49:01 2022

@author: eliza
"""

import matplotlib
import matplotlib as mpl
import math
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import readsav
import matplotlib.colors as colors
import random
import shapely.geometry as sg
from tfcat import TFCat
from os import path
import time as t
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']

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
    #print(time)
    #time = np.vectorize(fix_iso_format)(t_isostring)
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
    #print(time_view)
    return time_view, frequency, flux
    
def find_mask_aug(time_view_start, time_view_end, val, file_data,file_nolfe,polygon_fp,ind,fp_sav,augment_1,augment_2,augment_3,time_aug1, time_aug2,shift,side):
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

    
    if augment_1 == True:
        #print(file_nolfe)
        time_aug, freq_aug, s_aug = extract_data(file_nolfe, time_aug1, time_aug2, val)
        v2 = np.empty(s_aug.shape)
        v2[:] = np.nan
        v2 = np.concatenate([v2, v], axis=1)
        time_total=np.arange(len(time_aug)+ len(time_dt64))
    if augment_2 == True:
        time_aug, freq_aug, s_aug = extract_data(file_nolfe, time_aug1, time_aug2, val)
        v2 = np.empty(s_aug.shape)
        v2[:] = np.nan

        v2 = np.concatenate([v,v2], axis=1)
        time_total=np.arange(len(time_aug)+ len(time_dt64))
    if augment_3 ==True:
        w=len(time_dt64)
        h=len(frequency)
        #final array in total will be 6 hours long
        #3 minute res., 6hrs=120 timestamps
        final_w=200
        time_total=np.arange(final_w)
        #width of chopped LFE
        w_shifted = math.ceil(shift * w)
        #width to be added on
        extra_w =final_w-w_shifted
        #make empty arrays for filling up the rest of image
        extra_v = np.empty((h, extra_w))
        extra_v[:]=np.nan
        
        if side=='right':
            t_ = time_dt64[w_shifted]
            v_shifted=v[:,0:w_shifted]
            v2=np.concatenate([extra_v, v_shifted],axis=1)
        if side=='left':
            t_ = time_dt64[(w-w_shifted)]
            v_shifted=v[:,(w-w_shifted):]
            v2 = np.concatenate([v_shifted,extra_v],axis=1)
            
    w = len(time_total) 
    h = len(frequency)
    

    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)  
    ax.set_axis_off()
    cmap = mpl.colors.ListedColormap(['white']).copy()
    cmap.set_bad('black')
    ax.pcolormesh(time_total,frequency, v2,cmap=cmap,shading='auto')
    ax.set_axis_off()
    ax.set_yscale('log')
    figure_label = fp_sav + '/mask_images/mask_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    
    if augment_3 == True:
        return t_
    else:
        
        return None    
          
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

def plot_pol_and_flux(time_view_start, time_view_end, file,ind,fp_sav):
    
    time, freq, pol = extract_data(
        file, time_view_start=time_view_start, time_view_end=time_view_end,val='v')
    time, freq, flux = extract_data(
        file, time_view_start=time_view_start, time_view_end=time_view_end,val='s')
    
    #Polarization
    vmin =-1
    vmax = 1
    clrmap = 'binary_r'
    scaleZ = colors.Normalize(vmin=vmin, vmax=vmax)
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
    scaleZ = colors.LogNorm(vmin, vmax)
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

def plot_andaugment_pol_and_flux1(time_view_start, time_view_end, nolfe_start,nolfe_end, file,file2,ind,fp_sav,side=None):
    time, freq, pol = extract_data(
        file, time_view_start=time_view_start, time_view_end=time_view_end,val='v')
    time, freq, flux = extract_data(
        file, time_view_start=time_view_start, time_view_end=time_view_end,val='s')
    
    time_nolfe, freq_nolfe, pol_nolfe=extract_data(
        file2, time_view_start=nolfe_start, time_view_end=nolfe_end,val='v')
    time_nolfe, freq_nolfe, flux_nolfe=extract_data(
        file2, time_view_start=nolfe_start, time_view_end=nolfe_end,val='s')
    
    if side=='right':
        flux_total = np.concatenate([flux_nolfe, flux], axis=1)
        pol_total = np.concatenate([pol_nolfe, pol], axis=1)
        time_total = np.concatenate([time_nolfe, time],axis=0)
        time_total=np.arange(len(time_total))
    
    if side=='left':
        flux_total = np.concatenate([flux,flux_nolfe], axis=1)
        pol_total = np.concatenate([pol, pol_nolfe], axis=1)
        time_total = np.concatenate([time_nolfe, time],axis=0)
        time_total=np.arange(len(time_total))
    
    w = len(time_total) 
    h = len(freq)

    
    #Polarization
    vmin =-1
    vmax = 1
    scaleZ = colors.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    cmap = mpl.cm.get_cmap("binary_r").copy()
    ax.pcolormesh(time_total, freq, pol_total, norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = fp_sav + '/pol_images/pl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    
    w = len(time_total) 
    h = len(freq)
    #Flux
    vmin = 1e-25
    vmax = 1e-19
    scaleZ = colors.LogNorm(vmin, vmax)
    cmap = mpl.cm.get_cmap("binary_r").copy()
    cmap.set_bad('black')
    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax.set_axis_off()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    cmap = mpl.cm.get_cmap("binary_r").copy()
    cmap.set_bad('black')
    ax.pcolormesh(time_total, freq, flux_total, norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = fp_sav + '/flux_images/fl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)

    return None

def plot_andaugment_pol_and_flux2(time_view_start, time_view_end, file,fp_sav,ind,shift=None,side=None):
    
    time, freq, pol = extract_data(file, time_view_start=time_view_start, time_view_end=time_view_end,val='v')
    time, freq, flux = extract_data(
        file, time_view_start=time_view_start, time_view_end=time_view_end,val='s')
    #Shift flux and polarization arrays by specified amount
    w=len(time)
    h=len(freq)
    #final array in total will be 6 hours long
    #3 minute res., 6hrs=120 timestamps
    final_w=200
    time_total=np.arange(final_w)
    #width of chopped LFE
    w_shifted = math.ceil(shift * w)
    #width to be added on
    extra_w =final_w-w_shifted
    #make empty arrays for filling up the rest of image
    extra_p=np.zeros((h, extra_w))
    extra_f = np.empty((h, extra_w))
    extra_f[:]=np.nan
    
    if side=='right':
        pol_shifted=pol[:,0:w_shifted]
        flux_shifted=flux[:,0:w_shifted]
        final_f = np.concatenate([extra_f, flux_shifted],axis=1)
        final_p=np.concatenate([extra_p, pol_shifted],axis=1)
    if side=='left':
        pol_shifted=pol[:,(w-w_shifted):]
        flux_shifted=flux[:,(w-w_shifted):]
        final_f = np.concatenate([flux_shifted,extra_f],axis=1)
        final_p=np.concatenate([pol_shifted,extra_p],axis=1)


    
    #Polarization
    fig = plt.figure(frameon=False,figsize=(final_w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    vmin =-1
    vmax = 1
    clrmap = 'binary_r'
    scaleZ = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap(clrmap).copy()
    ax.pcolormesh(time_total, freq, final_p,norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = fp_sav + '/pol_images/pl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)

    
    #Flux
    fig = plt.figure(frameon=False,figsize=(final_w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    vmin = 1e-25
    vmax = 1e-19
    clrmap = 'binary_r'
    scaleZ = colors.LogNorm(vmin, vmax)
    cmap = mpl.cm.get_cmap(clrmap).copy()
    cmap.set_bad('black')
    ax.pcolormesh(time_total, freq, final_f,norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = fp_sav + '/flux_images/fl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    return None


'''____Type 1___'''

total_df=pd.read_csv(input_data_fp + "/total_timestamps.csv", parse_dates=['start','end'])
total_df_nosm = total_df.loc[total_df['label']!='LFE_sm', :].reset_index(drop=True)
lfe_df = total_df.loc[~total_df['label'].isin(['LFE_sm', 'NoLFE'])].reset_index(drop=True)
nolfe_df = pd.read_csv(input_data_fp + '/nolfe_timestamps.csv',parse_dates=['start','end'])


random.seed=42
len_aug1 = 50

sample_lfe=random.sample(list(np.arange(len(lfe_df))), len_aug1)
sample_nolfe=random.sample(list(np.arange(len(nolfe_df))), len_aug1)


nolfe_df_augment1=nolfe_df.loc[sample_nolfe, :].reset_index(drop=True)
lfe_df_augment1 = lfe_df.loc[sample_lfe,:].reset_index(drop=True)
lfe_df_augment1['label']=np.repeat('LFE_aug1', len(lfe_df_augment1))
lfe_df_augment1.to_csv(output_data_fp + '/ML_lfeaug1_timestamps.csv')


s_lfe=lfe_df_augment1['start']
e_lfe=lfe_df_augment1['end']

s_nolfe=nolfe_df_augment1['start']
e_nolfe = nolfe_df_augment1['end']
count=len(lfe_df) + len(nolfe_df)

for i, j, k, l in zip(s_lfe, e_lfe, s_nolfe, e_nolfe):
    year_lfe=datetime.strftime(i, '%Y')
    year_nolfe=datetime.strftime(k, '%Y')
    if year_lfe == '2017':
        file_lfe = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file_lfe = input_data_fp +'/SKR_{}_CJ.sav'.format(year_lfe)
        
    if year_nolfe == '2017':
        file_nolfe = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file_nolfe = input_data_fp + '/SKR_{}_CJ.sav'.format(year_nolfe)
    plt.ioff()
    polygon_fp = input_data_fp + '/SKR_LFEs.json'
    
    a = plot_andaugment_pol_and_flux1(i,j, k,l, file_lfe, file_nolfe, count, output_data_fp,side='right')
    b=find_mask_aug(time_view_start=i, time_view_end=j, val='s', file_data=file_lfe,file_nolfe=file_nolfe,\
                    polygon_fp=polygon_fp,ind=count,fp_sav=output_data_fp,augment_1=True,\
                    augment_2=False,augment_3=False,time_aug1=k, time_aug2=l,shift=None,side=None)
    print(count)
    count+=1

'''____Type 2___'''

len_aug2 = 50

random.seed = 10
sample_lfe = random.sample(list(np.arange(len(lfe_df))), len_aug2)
sample_nolfe = random.sample(list(np.arange(len(nolfe_df))), len_aug2)


nolfe_df_augment2 = nolfe_df.loc[sample_nolfe, :].reset_index(drop=True)
lfe_df_augment2 = lfe_df.loc[sample_lfe, :].reset_index(drop=True)
lfe_df_augment2['label'] = np.repeat('LFE_aug2', len(lfe_df_augment2))
lfe_df_augment2.to_csv(output_data_fp + '/ML_lfeaug2_timestamps.csv')

s_lfe = lfe_df_augment2['start']
e_lfe = lfe_df_augment2['end']

s_nolfe = nolfe_df_augment2['start']
e_nolfe = nolfe_df_augment2['end']
count = len(lfe_df) + len(nolfe_df) + len_aug1
for i, j, k, l in zip(s_lfe, e_lfe, s_nolfe, e_nolfe):
    year_lfe=datetime.strftime(i, '%Y')
    year_nolfe=datetime.strftime(k, '%Y')
    if year_lfe == '2017':
        file_lfe = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file_lfe = input_data_fp + '/SKR_{}_CJ.sav'.format(year_lfe)
    if year_nolfe == '2017':
        file_nolfe = input_data_fp + 'SKR_2017_001-258_CJ.sav'
    else: 
        file_nolfe = input_data_fp + '/SKR_{}_CJ.sav'.format(year_nolfe)
    plt.ioff()
    a=plot_andaugment_pol_and_flux1(time_view_start=i, time_view_end=j, nolfe_start=k,nolfe_end=l, 
                                  file=file_lfe,file2=file_nolfe,ind=count,fp_sav=output_data_fp,side='left')
    b=find_mask_aug(time_view_start=i, time_view_end=j, val='s', file_data=file_lfe,file_nolfe=file_nolfe,\
                    polygon_fp=polygon_fp,ind=count,fp_sav=output_data_fp,augment_1=False,\
                    augment_2=True,augment_3=False,time_aug1=k, time_aug2=l,shift=None,side=None)
    print(count)
    count+=1

'''____Type 3___'''

lfe_df['dur'] = lfe_df.apply(lambda x: (x.end-x.start).total_seconds()/3600,axis=1)

len_aug3and4 = 100

test_df1 = lfe_df.loc[(lfe_df['dur']>8)&(lfe_df['dur']<12)&(lfe_df['label']!='LFE_sm')
                    ,:].reset_index(drop=True)
test_df1 = test_df1.loc[:len_aug3and4-1,:]
split = math.ceil(len(test_df1)/2)
shifts = np.arange(0.45, 0.7, 0.05)

len_aug3 = split

lfe_df_augment3 = test_df1.loc[0:split-1,:].reset_index(drop=True)
picked_shifts = []
count=len(lfe_df) + len(nolfe_df) + len_aug1 +len_aug2

t_shifted = []
for i,j in zip(lfe_df_augment3['start'], lfe_df_augment3['end']):
    year_lfe=datetime.strftime(i, '%Y')
    
    if year_lfe == '2017':
        file_lfe = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file_lfe = input_data_fp + '/SKR_{}_CJ.sav'.format(year_lfe)
    shift=random.choice(shifts)
    picked_shifts.append(shift)
    a = plot_andaugment_pol_and_flux2(i,j, file_lfe, output_data_fp,count,shift,side='left')
    t_start=find_mask_aug(time_view_start=i, time_view_end=j, val='s', file_data=file_lfe,file_nolfe=None,\
                    polygon_fp=polygon_fp,ind=count,fp_sav=output_data_fp,augment_1=False,\
                    augment_2=False,augment_3=True,time_aug1=None, time_aug2=None,shift=shift,side='left')
    t_shifted.append(t_start)
    print(count)
    count+=1
lfe_df_augment3['shift']=picked_shifts
lfe_df_augment3['start'] = t_shifted
lfe_df_augment3.to_csv(output_data_fp + '/ML_lfeaug3_timestamps.csv',index=False)


''' Type 4'''

len_aug4 = split
lfe_df_augment4 = test_df1.loc[split:,:].reset_index(drop=True)
count = len(lfe_df) + len(nolfe_df) + len_aug1 +len_aug2 +len_aug3
picked_shifts=[]
t_shifted = []
for i,j in zip(lfe_df_augment4['start'], lfe_df_augment4['end']):
    year_lfe=datetime.strftime(i, '%Y')
    if year_lfe == '2017':
        file_lfe = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file_lfe = input_data_fp + '/SKR_{}_CJ.sav'.format(year_lfe)
    shift=random.choice(shifts)
    picked_shifts.append(shift)
    a = plot_andaugment_pol_and_flux2(i,j, file_lfe, output_data_fp, count,shift,side='right')
    t_end=find_mask_aug(time_view_start=i, time_view_end=j, val='s', file_data=file_lfe,file_nolfe=None,\
                    polygon_fp=polygon_fp,ind=count,fp_sav=output_data_fp, augment_1=False,\
                    augment_2=False, augment_3=True,time_aug1=None, time_aug2=None,shift=shift,side='right')
    t_shifted.append(t_end)
    print(count)
    count+=1
    
lfe_df_augment4['shift']=picked_shifts
lfe_df_augment4['start'] = t_shifted
lfe_df_augment4.to_csv(output_data_fp + '/ML_lfeaug4_timestamps.csv',index=False)


'''____Type 5___Multiple LFEs in one image'''

lfe_df_06 = lfe_df.loc[lfe_df['start'].between(pd.Timestamp('20060101'), pd.Timestamp('20070101')),:].reset_index(drop=True)

lfe_df_dec = lfe_df.loc[lfe_df['start'].between(pd.Timestamp('20081201'), pd.Timestamp('20090101')),:].reset_index(drop=True)
lfe_df_mar= lfe_df.loc[lfe_df['start'].between(pd.Timestamp('20080301'), pd.Timestamp('20080401')),:].reset_index(drop=True)
lfe_df_may= lfe_df.loc[lfe_df['start'].between(pd.Timestamp('20080501'), pd.Timestamp('20080601')),:].reset_index(drop=True)
df_fully_seg = pd.concat([lfe_df_06, lfe_df_mar, lfe_df_may, lfe_df_dec],axis=0).sort_values(by='start').reset_index(drop=True)


diff = [(i-j).total_seconds()/3600 for i,j in zip(df_fully_seg['start'].iloc[1:],df_fully_seg['end'].iloc[:-1])]
diff.append(np.nan)
df_fully_seg['diff'] = diff
df_aug4 = df_fully_seg.loc[(df_fully_seg['diff']<4)&(df_fully_seg['diff']>3),:]

#split into groups for plotting
inds = df_aug4.index
inds_items = [i+1 for i in inds]
final_inds = np.unique(np.sort(np.concatenate([inds, inds_items])))
df_aug5 = df_fully_seg.loc[final_inds,['start','end']].reset_index(drop=True)
diff = [(i-j).total_seconds()/3600 for i,j in zip(df_aug5['start'].iloc[1:],df_aug5['end'].iloc[:-1])]
diff.append(np.nan)
df_aug5['diff']=diff
split = [i+1 for i,j in enumerate(diff) if j> 4]
split = np.unique(split)
groups = np.split(final_inds, split)
start_inds = [i[0] for i in groups]
end_inds = [i[-1] for i in groups]
starts = df_fully_seg.loc[start_inds,'start'].reset_index(drop=True)
ends = df_fully_seg.loc[end_inds,'end'].reset_index(drop=True)
lfe_df_augment5 = pd.DataFrame({'start':starts, 'end':ends})
lfe_df_augment5.to_csv(output_data_fp + '/ML_lfeaug5_timestamps.csv',index=False)

count = len(lfe_df) + len(nolfe_df) + len_aug1 +len_aug2 +len_aug3+len_aug4
for i,j in zip(starts,ends):
    year_lfe=datetime.strftime(i, '%Y')
    if year_lfe == '2017':
        file_lfe = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file_lfe = input_data_fp + '/SKR_{}_CJ.sav'.format(year_lfe)
    
    plt.ioff()
    a = plot_pol_and_flux(i, j, file_lfe, count, output_data_fp)
    b=find_mask(i, j, 's', file_lfe, polygon_fp, count, output_data_fp)
    print(count)
    count+=1

