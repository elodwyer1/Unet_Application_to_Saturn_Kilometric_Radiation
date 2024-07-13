# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:35:19 2021

@author: eliza
"""
import matplotlib.ticker as mticker 
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from bisect import bisect_left
import configparser
from datetime import datetime
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']

#this needs to match up to bins and step size defined in 'Sliding_Window_Code.py'
step=1
bins = np.arange(0,24.1,step)
#0 if step size is 1, 1 if step size is 0.1, 2 if 0.01.
decimals=0

def load_ephem(dtime,dtime2):
    year=pd.to_datetime(dtime).year
    fp= 'C:/Users/eliza/Desktop/Python_Scripts/output_data/trajectory{}.csv'.format(year)
    orbit_df = pd.read_csv(fp, parse_dates=['datetime_ut'])
    orbit_df = orbit_df.loc[orbit_df['datetime_ut'].between(dtime,dtime2), :]
    return orbit_df
def flux_norm(dtime,dtime2):
    bins_ = list(bins)
    orbit_df = load_ephem(dtime, dtime2)
    if len(orbit_df) !=0:
        LT_val1 = round(orbit_df['localtime'].iloc[0],decimals)
        LT_val2 = round(orbit_df['localtime'].iloc[-1],decimals)
    
        LT1=bins_.index(LT_val1)
        LT2=bins_.index(LT_val2)
     
        lat_arr=np.array(orbit_df['lat'].unique())
        mean_lat=np.take(lat_arr, lat_arr.size//2) 
        
        if mean_lat > -5 and mean_lat <5:
            flux_arr = np.load('C:/Users/eliza/Desktop/Python_Scripts/output_data/lowlat_flux.npy', allow_pickle=True)
        else:
            flux_arr= np.load('C:/Users/eliza/Desktop/Python_Scripts/output_data/highlat_flux.npy', allow_pickle=True)
        
        if LT2 <LT1:
            lt1_arr = np.arange(LT1, len(bins), step)
            lt2_arr = np.arange(0, LT2+step, step)
            lt_arr = np.concatenate([lt1_arr, lt2_arr])
            lt_arr = np.sort(lt_arr)
            flux_arr = flux_arr[lt_arr]
            ttl_flux_arr=np.concatenate(flux_arr)
        elif LT1==LT2:
            ttl_flux_arr = flux_arr[LT1]
            
        else:
            lt_arr = np.arange(LT1, LT2+step, step)
            flux_arr = flux_arr[lt_arr]
            ttl_flux_arr=np.concatenate(flux_arr)
    else:
        flux_arr1 = np.load('C:/Users/eliza/Desktop/Python_Scripts/output_data/lowlat_flux.npy', allow_pickle=True)
        flux_arr1 = np.concatenate(flux_arr1, axis=0)
        flux_arr2= np.load('C:/Users/eliza/Desktop/Python_Scripts/output_data/highlat_flux.npy', allow_pickle=True)
        flux_arr2 = np.concatenate(flux_arr2, axis=0)
        ttl_flux_arr = np.concatenate([flux_arr1, flux_arr2], axis=0)
    p80 = np.percentile(ttl_flux_arr, 80)
    p10 = np.percentile(ttl_flux_arr, 10)
    return p10, p80
#From A.R Fogg :)
def median_absolute_deviation(data):
    # Calculates the median absolution deviation of the sample
    # Returns the MAD param, and the median
    median=np.nanmedian(data)
    
    difs=np.empty(data.size)
    difs[:]=np.nan
    for i in range(data.size):
        difs[i]=abs(data[i]-median)

    mad=np.nanmedian(difs)
    mad = round(mad, 2)
    median = round(median,2)
    return mad,median


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0
    if pos == len(myList):
        return -1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos-1


def ephemeris_labels(dtime):
    """
    Given a `datetime`, return the radial distance (in Earth radii), the GSE
    latitude and the GSE local time that corresponds to the closest recorded
    position of Wind at that time.

    Filename required to be specified by datetime 
    """

    year=pd.to_datetime(dtime).year
    fp= 'C:/Users/eliza/Desktop/Python_Scripts/output_data/trajectory{}.csv'.format(year)
    orbit_df = pd.read_csv(fp,parse_dates=['datetime_ut'])
    dtime_pandas=pd.Timestamp(dtime)
    orbit_df.index=orbit_df['datetime_ut']
    dtime_ind = orbit_df.index.get_indexer([dtime_pandas],method='nearest')
    dtime_val=orbit_df.index[dtime_ind]
    #dtime_ind=take_closest(orbit_df['datetime_ut'], dtime_pandas)
    orbit_df = orbit_df.loc[dtime_val, :].reset_index(drop=True)
    
    # replace these with appropriate data headers etc
    dist = orbit_df.loc[0,'range']
    lat = orbit_df.loc[0,'lat']
    lt = orbit_df.loc[0,'localtime']

    eph_strs = [str(x) for x in [dist, lat, lt]]
    
    return eph_strs

@mticker.FuncFormatter
def ephemeris_fmt_hour_tick(tick_val,_):
    """
    Call with eg

        ax.xaxis.set_major_formatter(plt.FuncFormatter(ephemeris_fmt))

    or, if decorator @matplotlib.ticker.FuncFormatter used 
    
        ax.xaxis.set_major_formatter(ephemeris_fmt)
        
    """
    
    # Convert matplotlib datetime float to date
    
    tick_dt=mdates.num2date(tick_val)
    tick_dt = tick_dt.replace(tzinfo=None)

    #s_doy = tick_dt.timetuple().tm_yday 
    #s_h =  (tick_dt-pd.Timestamp(datetime.strftime(tick_dt, '%Y-%m-%d'))).total_seconds()/86400
    #s_h = round(s_h, 3)
    #tick_str = str(s_doy + s_h)
    tick_str = datetime.strftime(tick_dt, ('%m-%d %H:%M'))
    tick_str = tick_str.replace(' ', '\n')
    # this returns corresponding radial dist, gse_lat, gse_lt for the tick
    # as strings in a list
    eph_str = ephemeris_labels(tick_dt)
    eph_str = [tick_str] + eph_str
    tick_str = '\n'.join(eph_str)

    return tick_str
