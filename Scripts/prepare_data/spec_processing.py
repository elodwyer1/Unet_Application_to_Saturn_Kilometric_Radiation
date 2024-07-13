import pandas as pd
import numpy as np
from datetime import datetime
from scipy.io import readsav
from tfcat import TFCat
import shapely.geometry as sg
from os import path
import matplotlib as mpl
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..read_config import config
from os import (path, makedirs)
mpl.use('Agg')  # Use 'Agg' backend for environments without GUI support

def read_radio(time_view_start: datetime, time_view_end: datetime, val: str):
    time_index = 't'
    freq_index = 'f'
    val_index = val
    year = time_view_start.year
    if year != 2017:
        radio_data = readsav(config.input_data_fp + f'/SKR_{year}_CJ.sav')
    else:
        radio_data = readsav(config.input_data_fp + f'/SKR_2017_001-258_CJ.sav')
    time_dt64 = convert_radio_doyfrac_to_dt64(radio_data[time_index])
    time_view_dt64 = time_dt64[(time_dt64 >= time_view_start) & (time_dt64 < time_view_end)]
    s = radio_data[val_index][:, (time_dt64 >= time_view_start) & (time_dt64 < time_view_end)].copy()
    frequency_tmp = radio_data[freq_index].copy()
    return s, frequency_tmp, time_view_dt64

    
def doy_to_dt64(doy_frac, start_year=1997):
    # Calculate the base date as the last day of the previous year
    base_date = np.datetime64(f'{start_year - 1}-12-31')
    
    # Separate the integer part and the fractional part of the day of year
    integer_days = int(doy_frac)
    fractional_days = doy_frac - integer_days
    
    # Calculate the target date by adding the integer days
    target_date = base_date + np.timedelta64(integer_days, 'D')
    
    # Calculate the seconds from the fractional part
    fractional_seconds = int(fractional_days * 24 * 60 * 60)
    
    # Add the fractional seconds to the target date
    target_date += np.timedelta64(fractional_seconds, 's')
    
    return target_date

def convert_radio_doyfrac_to_dt64(t_doy):
    t_doy_copy = t_doy.copy()
    vectorized_doy_to_dt64 = np.vectorize(doy_to_dt64)
    t_dt64 = vectorized_doy_to_dt64(t_doy_copy)
    return t_dt64

def interp(s, frequency_tmp, time_view):
    step = (np.log10(max(frequency_tmp)) - np.log10(min(frequency_tmp))) / 383
    frequency = 10 ** (np.arange(np.log10(frequency_tmp[0]), np.log10(frequency_tmp[-1]) + step / 2, step,
                                  dtype=float))
    flux = np.zeros((frequency.size, len(time_view)), dtype=float)
    for i in range(len(time_view)):
        flux[:, i] = np.interp(frequency, frequency_tmp, s[:, i])
    return flux, frequency

def extract_data(time_view_start: datetime, time_view_end: datetime, val: str):
    s, frequency_tmp, time_view = read_radio(time_view_start, time_view_end, val)
    flux, frequency = interp(s, frequency_tmp, time_view)
    return time_view, frequency, flux

def get_polygons(start: datetime, end: datetime):
    unix_start = start.timestamp()
    unix_end = end.timestamp()
    polygon_array = []
    if path.exists(config.input_data_fp + '/SKR_LFEs.json'):
        catalogue = TFCat.from_file(config.input_data_fp + '/SKR_LFEs.json')
        for i in range(len(catalogue)):
            if catalogue._data.features[i].properties['feature_type'] != 'LFE_sm':
                time_points = np.array(catalogue._data.features[i]['geometry']['coordinates'][0])[:, 0]
                if any(time_points <= unix_end) and any(time_points >= unix_start):
                    polygon_array.append(catalogue._data.features[i]['geometry']['coordinates'][0])
    return polygon_array

def plot_spec(time_dt64, frequency, v):
    #width and height of array (f,t)
    w = len(time_dt64) 
    h = len(frequency)

    fig = plt.figure(frameon=False, figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)  
    ax.set_axis_off()
    cmap = mpl.colors.ListedColormap(['white']).copy()
    cmap.set_bad('black')
    ax.pcolormesh(time_dt64,frequency, v, cmap=cmap, shading='auto')
    ax.set_axis_off()
    ax.set_yscale('log')
    return ax, fig

def save_mask(fig, ind):
    figure_label = config.output_data_fp + '/mask_images/mask_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)

def find_emptymask(time_view_start: datetime, time_view_end: datetime, val: str):
    
    #signal data and time frequency values within the time range specified.
    time_view_dt64, frequency, flux = extract_data(time_view_start, time_view_end, val)
    
    v = np.empty(flux.shape)
    v[:] = np.nan
    
    return time_view_dt64, frequency, v


def find_mask(time_view_start, time_view_end, val):
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval
    polygon_array=get_polygons(time_view_start, time_view_end)
    #signal data and time frequency values within the time range specified.
    time_dt64, frequency, flux=extract_data(time_view_start, time_view_end, val)
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
    #Make figure
    return time_dt64, frequency, v

def plot_mask(time_view_start, time_view_end, val, ind):
    time_dt64, frequency, v = find_mask(time_view_start, time_view_end, val)
    ax, fig = plot_spec(time_dt64, frequency, v)
    save_mask(fig, ind)
    plt.close(fig)

def plot_empty_mask(time_view_start: datetime, time_view_end: datetime, val: str,
                    ind: int):
    time_view_dt64, frequency, v = find_emptymask(time_view_start, time_view_end, val,
                    ind)
    #Make figure
    ax, fig = plot_spec(time_view_dt64, frequency, v)
    save_mask(fig, ind)
    plt.close(fig)
    

def plot_pol_and_flux(time_view_start, time_view_end, ind):
    time, freq, pol = extract_data(
        time_view_start=time_view_start, time_view_end=time_view_end,val='v')
    time, freq, flux = extract_data(
        time_view_start=time_view_start, time_view_end=time_view_end,val='s')
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
    figure_label = config.output_data_fp + '/pol_images/pl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    plt.clf()
    #Flux
    vmin = 1e-25
    vmax = 1e-19
    scaleZ = mpl.colors.LogNorm(vmin, vmax)
    cmap = plt.get_cmap("binary_r").copy()
    cmap.set_bad('black')
    fig = plt.figure(frameon=False,figsize=(w/100,h/100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.pcolormesh(time, freq, flux, norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_axis_off()
    ax.set_yscale('log')
    figure_label = config.output_data_fp + '/flux_images/fl_img' +str(ind).zfill(4)+'.png'
    fig.savefig(figure_label)
    plt.close(fig)
    return None
def tzip(*args, **kwargs):
    """
    Wrap zip with tqdm to create a progress bar for multiple iterables.
    """
    total = kwargs.pop('total', None)  # Allow total to be passed or default to None
    # Create a generator that uses tqdm to wrap zip
    for item in tqdm(zip(*args), total=total, **kwargs):
        yield item

def make_train_specs():
    polygon_fp= config.input_data_fp + "/SKR_LFEs.json"
    #Load start and end times of LFEs and non-LFEs
    total_df=pd.read_csv(config.output_data_fp + "/total_timestamps.csv",
                          parse_dates=['start','end'])

    total_df_nosm = total_df.loc[total_df['label']!='LFE_sm', :].reset_index(drop=True)

    lfe_df = total_df.loc[~total_df['label'].isin(['LFE_sm', 'NoLFE'])].reset_index(drop=True)

    nolfe_df = total_df.loc[total_df['label']=='NoLFE', :].reset_index(drop=True)

    #make path to save flux, polarization and mask images
    if not path.exists(config.output_data_fp  + '/flux_images/'):
        makedirs(config.output_data_fp + '/flux_images/')
    if not path.exists(config.output_data_fp + '/pol_images/'):
        makedirs(config.output_data_fp + '/pol_images/')
    if not path.exists(config.output_data_fp + '/mask_images/'):
        makedirs(config.output_data_fp + '/mask_images/')
    
    print('\n Making LFE dataset.')
    #make LFE images
    with tqdm(total=len(lfe_df)) as pbar:
        for day1, day2, i in tzip(lfe_df['start'], lfe_df['end'], lfe_df.index):
            plt.ioff()
            #make flux and polarization spectrograms and save as images.
            plot_pol_and_flux(day1, day2,i)
            val='s'
            #make corresponding masked spectrogram and save as image.
            plot_mask(day1, day2, val, i) 
            pbar.update(1)

    #Make noLFE images
    print('\n Making non-LFE dataset.')
    nolfe_index = np.arange(len(lfe_df), len(total_df_nosm))
    with tqdm(total=len(nolfe_index)) as pbar:
        for day1, day2, i in tzip(nolfe_df['start'], nolfe_df['end'],
                                    nolfe_index):
            plt.ioff()
            #make flux and polarization spectrograms and save as images.
            plot_pol_and_flux(day1, day2, i)
            val='s'
            #make corresponding masked spectrogram and save as image.
            plot_emptymask(day1, day2, val, i) 
            pbar.update(1)