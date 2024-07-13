# -*- coding: utf-8 -*-

from tqdm import tqdm
import matplotlib as mpl
import math
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import shapely.geometry as sg
import time as t
mpl.use('Agg')  # Use 'Agg' backend for environments without GUI support
from ..read_config import config
from .spec_processing import (read_radio, extract_data, plot_pol_and_flux,
                            find_emptymask, find_mask, save_mask, tzip, 
                            get_polygons, plot_spec, plot_mask)
    
def find_mask_aug(time_view_start, time_view_end, val,ind,augment_1,augment_2,augment_3,time_aug1, time_aug2,shift,side):
    time_dt64, frequency, v = find_mask(time_view_start, time_view_end, val)
    
    if augment_1 == True:
        time_aug, freq_aug, s_aug = extract_data(time_aug1, time_aug2, val)
        v2 = np.empty(s_aug.shape)
        v2[:] = np.nan
        v2 = np.concatenate([v2, v], axis=1)
        time_total=np.arange(len(time_aug)+ len(time_dt64))
    if augment_2 == True:
        time_aug, freq_aug, s_aug = extract_data(time_aug1, time_aug2, val)
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
            

    ax, fig = plot_spec(time_total,frequency, v2)
    save_mask(fig, ind)
    

    if augment_3 == True:
        return t_
    else:
        
        return None    
          




def plot_andaugment_pol_and_flux1(time_view_start, time_view_end, nolfe_start,nolfe_end, ind, side=None):
    time, freq, pol = extract_data(
        time_view_start=time_view_start, time_view_end=time_view_end,val='v')
    time, freq, flux = extract_data(
        time_view_start=time_view_start, time_view_end=time_view_end,val='s')
    
    time_nolfe, freq_nolfe, pol_nolfe=extract_data(
        time_view_start=nolfe_start, time_view_end=nolfe_end,val='v')
    time_nolfe, freq_nolfe, flux_nolfe=extract_data(
        time_view_start=nolfe_start, time_view_end=nolfe_end,val='s')
    
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
    cmap = plt.get_cmap("binary_r").copy()
    ax.pcolormesh(time_total, freq, pol_total, norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = config.output_data_fp + '/pol_images/pl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    
    w = len(time_total) 
    h = len(freq)
    #Flux
    vmin = 1e-25
    vmax = 1e-19
    scaleZ = colors.LogNorm(vmin, vmax)
    cmap = plt.get_cmap("binary_r").copy()
    cmap.set_bad('black')
    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax.set_axis_off()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    cmap = plt.get_cmap("binary_r").copy()
    cmap.set_bad('black')
    ax.pcolormesh(time_total, freq, flux_total, norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = config.output_data_fp + '/flux_images/fl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)

    return None

def plot_andaugment_pol_and_flux2(time_view_start, time_view_end, ind,shift=None,side=None):
    
    time, freq, pol = extract_data(time_view_start=time_view_start, time_view_end=time_view_end,val='v')
    time, freq, flux = extract_data(
        time_view_start=time_view_start, time_view_end=time_view_end,val='s')
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
    cmap = plt.get_cmap(clrmap).copy()
    ax.pcolormesh(time_total, freq, final_p,norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = config.output_data_fp + '/pol_images/pl_img' +str(ind).zfill(4)+'.png'
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
    cmap = plt.get_cmap(clrmap).copy()
    cmap.set_bad('black')
    ax.pcolormesh(time_total, freq, final_f,norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    figure_label = config.output_data_fp + '/flux_images/fl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    return None


def make_augment_data():
    # type 1
    
    total_df=pd.read_csv(config.output_data_fp + "/total_timestamps.csv", parse_dates=['start','end'])
    lfe_df = total_df.loc[~total_df['label'].isin(['LFE_sm', 'NoLFE'])].reset_index(drop=True)
    nolfe_df = total_df.loc[total_df['label']=='NoLFE', :].reset_index(drop=True)


    random.seed=42
    len_aug1 = 50
    
    sample_lfe=random.sample(list(np.arange(len(lfe_df))), len_aug1)
    sample_nolfe=random.sample(list(np.arange(len(nolfe_df))), len_aug1)


    nolfe_df_augment1=nolfe_df.loc[sample_nolfe, :].reset_index(drop=True)
    lfe_df_augment1 = lfe_df.loc[sample_lfe,:].reset_index(drop=True)
    lfe_df_augment1['label']=np.repeat('LFE_aug1', len(lfe_df_augment1))
    lfe_df_augment1.to_csv(config.output_data_fp + '/ML_lfeaug1_timestamps.csv')


    s_lfe=lfe_df_augment1['start']
    e_lfe=lfe_df_augment1['end']

    s_nolfe=nolfe_df_augment1['start']
    e_nolfe = nolfe_df_augment1['end']
    count=len(lfe_df) + len(nolfe_df)
    print('Augment type 1')

    with tqdm(total=len_aug1) as pbar:
        for start_lfe, end_lfe, start_nolfe, end_nolfe in zip(s_lfe, e_lfe, s_nolfe, e_nolfe):
            
            plot_andaugment_pol_and_flux1(start_lfe, end_lfe, start_nolfe, end_nolfe,count, side='right')
            find_mask_aug(start_lfe, end_lfe, val='s',\
                            ind=count,augment_1=True,\
                            augment_2=False,augment_3=False,time_aug1=start_nolfe, time_aug2=end_nolfe,shift=None,side=None)
            count+=1
            pbar.update(1)
     
    # type 2

    len_aug2 = 50
    
    random.seed = 10
    sample_lfe = random.sample(list(np.arange(len(lfe_df))), len_aug2)
    sample_nolfe = random.sample(list(np.arange(len(nolfe_df))), len_aug2)
    

    nolfe_df_augment2 = nolfe_df.loc[sample_nolfe, :].reset_index(drop=True)
    lfe_df_augment2 = lfe_df.loc[sample_lfe, :].reset_index(drop=True)
    lfe_df_augment2['label'] = np.repeat('LFE_aug2', len(lfe_df_augment2))
    lfe_df_augment2.to_csv(config.output_data_fp + '/ML_lfeaug2_timestamps.csv')

    s_lfe = lfe_df_augment2['start']
    e_lfe = lfe_df_augment2['end']

    s_nolfe = nolfe_df_augment2['start']
    e_nolfe = nolfe_df_augment2['end']
    count = len(lfe_df) + len(nolfe_df) + len_aug1

    print('Augment type 2')
    with tqdm(total=len_aug2) as pbar:
        for start_lfe, end_lfe, start_nolfe, end_nolfe in zip(s_lfe, e_lfe, s_nolfe, e_nolfe):
        
            plt.ioff()
            plot_andaugment_pol_and_flux1(start_lfe, end_lfe, start_nolfe, end_nolfe,
                                        ind=count,side='left')
            find_mask_aug(start_lfe, end_lfe, val='s',
                        ind=count,augment_1=False,\
                            augment_2=True,augment_3=False,time_aug1=start_nolfe, time_aug2=end_nolfe,shift=None,side=None)
            count+=1
            pbar.update(1)
    
    #type 3

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
    
    print('Augment type 3')
    with tqdm(total=len_aug3) as pbar:
        for start_lfe, end_lfe in zip(lfe_df_augment3['start'], lfe_df_augment3['end']):
            shift=random.choice(shifts)
            picked_shifts.append(shift)
            plot_andaugment_pol_and_flux2(start_lfe, end_lfe,count,shift,side='left')
            t_start=find_mask_aug(time_view_start=start_lfe, time_view_end=end_lfe, val='s',\
                            ind=count, augment_1=False,\
                            augment_2=False,augment_3=True,time_aug1=None, time_aug2=None,shift=shift,side='left')
            t_shifted.append(t_start)
            count+=1
            pbar.update(1)
            
    lfe_df_augment3['shift']=picked_shifts
    lfe_df_augment3['start'] = t_shifted
    lfe_df_augment3.to_csv(config.output_data_fp + '/ML_lfeaug3_timestamps.csv',index=False)
    
    
    #TYPE 4
    
    len_aug4 = split
    
    lfe_df_augment4 = test_df1.loc[split:,:].reset_index(drop=True)
    count = len(lfe_df) + len(nolfe_df) + len_aug1 +len_aug2 +len_aug3
    picked_shifts=[]
    t_shifted = []
    print('Augment type 4')
    
    with tqdm(total=len_aug4) as pbar:
        for start_lfe, end_lfe in zip(lfe_df_augment4['start'], lfe_df_augment4['end']):

            shift=random.choice(shifts)
            picked_shifts.append(shift)
            plot_andaugment_pol_and_flux2(start_lfe, end_lfe, count,shift,side='right')
            t_end=find_mask_aug(start_lfe, end_lfe, val='s',
                            ind=count, augment_1=False,\
                            augment_2=False, augment_3=True,time_aug1=None, time_aug2=None,shift=shift,side='right')
            t_shifted.append(t_end)
            count+=1
            pbar.update(1)
        
    lfe_df_augment4['shift']=picked_shifts
    lfe_df_augment4['start'] = t_shifted
    lfe_df_augment4.to_csv(config.output_data_fp + '/ML_lfeaug4_timestamps.csv',index=False)
    
    
    # type 5
    
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
    lfe_df_augment5.to_csv(config.output_data_fp + '/ML_lfeaug5_timestamps.csv',index=False)

    count = len(lfe_df) + len(nolfe_df) + len_aug1 +len_aug2 +len_aug3+len_aug4
    print('Augment type 5')
    with tqdm(total=len(lfe_df_augment5)) as pbar:
        for start_lfe, end_lfe in zip(starts,ends):
            plt.ioff()
            plot_pol_and_flux(start_lfe, end_lfe, count)
            plot_mask(start_lfe, end_lfe,'s', count)
            count+=1
            pbar.update(1)
            
    

    