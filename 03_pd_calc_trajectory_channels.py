# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:34:39 2022

@author: eliza
"""
import pandas as pd
import numpy as np
from datetime import datetime
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']


def return_trajectory_chunk(start, end):
    year= datetime.strftime(start, '%Y')
    traj_df =pd.read_csv(input_data_fp +'/trajectory{}.csv'.format(year), parse_dates=['datetime_ut']) 
    traj_df_chunk=traj_df.loc[traj_df['datetime_ut'].between(start, end),:].reset_index(drop=True)
    if len(traj_df_chunk) == 0:
        traj_df_chunk = traj_df.copy()
    return traj_df_chunk
def interpolate_traj(start, end):
    #use nearest neighbour interpolation, same as used in the image resizing.

    #print(start)
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
    print(start)
    return lt_med, lt_std, lat_med, lat_std

''''Make Catalogue (start, stop, type),trajectory info'''
df=pd.read_csv(input_data_fp + "/total_timestamps.csv", parse_dates=['start','end'])
df_nosm = df.loc[df['label']!='LFE_sm', :].reset_index(drop=True)
df_aug1=pd.read_csv(output_data_fp + '/ML_lfeaug1_timestamps.csv',parse_dates=['start','end'],index_col=False)
df_aug1=df_aug1.loc[:,['start','end','label']]
df_aug2=pd.read_csv(output_data_fp + '/ML_lfeaug2_timestamps.csv',parse_dates=['start','end'],index_col=False)
df_aug2=df_aug2.loc[:,['start','end','label']]
df_aug3=pd.read_csv(output_data_fp + '/ML_lfeaug3_timestamps.csv',parse_dates=['start','end'],index_col=False).loc[:,['start','end']]
df_aug3['label']=np.repeat('LFE_aug3',len(df_aug3))
df_aug4=pd.read_csv(output_data_fp + '/ML_lfeaug4_timestamps.csv',parse_dates=['start','end'],index_col=False).loc[:,['start','end']]
df_aug4['label']=np.repeat('LFE_aug4',len(df_aug4))
df_aug5=pd.read_csv(output_data_fp + '/ML_lfeaug5_timestamps.csv',parse_dates=['start','end'],index_col=False).loc[:,['start','end']]
df_aug5['label']=np.repeat('LFE_aug5',len(df_aug5))
total_df=pd.concat([df_nosm, df_aug1, df_aug2,df_aug3, df_aug4, df_aug5],axis=0)
total_df.to_csv(output_data_fp + '/ML_total_timestamps_withaug.csv',index=False)


vals = np.array(list(map(take_median_std,total_df['start'] , total_df['end'])))
total_df['lt_median']=vals[:,0]
total_df['lt_stdev']=vals[:,1]
total_df['lat_median']=vals[:,2]
total_df['lat_stdev']=vals[:,3]

total_df.to_csv(output_data_fp + '/ML_total_catalogue_withaug.csv',index=False)
