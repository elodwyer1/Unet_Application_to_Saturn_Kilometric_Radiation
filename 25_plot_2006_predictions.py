# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:51:35 2022

@author: eliza
"""

import numpy as np
from scipy.io import readsav
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits import axes_grid1
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import time as t
import pandas as pd
from tfcat import TFCat
import shapely.geometry as sg
from os import path, rename
import matplotlib as mpl
import plotting_func as plt_func
import configparser
import matplotlib.image as mpimg
import calendar
import os
from astropy.time import Time
config = configparser.ConfigParser()
import warnings
warnings.filterwarnings("ignore", 'No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.')
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']

def legend_without_duplicate_labels(ax, loc_):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend_properties = {'weight':'bold'}
    ax.legend(*zip(*unique), loc=loc_,fontsize=14,labelcolor='orangered',fancybox=True)
    

'''____This was adapted from original code for the space labelling tool!!!___'''
def get_polygons(polygon_fp,start, end):
    unix_start=t.mktime(start.utctimetuple())
    unix_end=t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array=[]
    if path.exists(polygon_fp):
        catalogue = TFCat.from_file(polygon_fp)
        for i in range(len(catalogue._data['features'])):
                time_points=np.array(catalogue._data['features'][i]['geometry']['coordinates'][0])[:,0]
                if any(time_points <= unix_end) and any(time_points >= unix_start):
                    polygon_array.append(np.array(catalogue._data['features'][i]['geometry']['coordinates'][0]))
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval         
    return polygon_array

'''____This was adapted from original code for the space labelling tool!!!___'''
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
    s = file[val_index][:, (time >= time_view_start) & (time <= time_view_end)].copy()
    frequency_tmp = file[freq_index].copy()

    # frequency_tmp is in log scale from f[0]=3.9548001 to f[24] = 349.6542
    # and then in linear scale above so it's needed to transfrom the frequency
    # table in a full log table and einterpolate the flux table (s --> flux
    frequency = 10**(np.arange(np.log10(frequency_tmp[0]), np.log10(frequency_tmp[-1]), (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/399, dtype=float))
    flux = np.zeros((frequency.size, len(time_view)), dtype=float)

    for i in range(len(time_view)):
        flux[:, i] = np.interp(frequency, frequency_tmp, s[:, i])

    return time_view, frequency, flux

def plot_mask(ax,time_view_start, time_view_end, val, file_data,polygon_fp,colour_in=None):
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
    
    #colorbar limits
    #vmin = np.quantile(flux[flux > 0.], 0.05)
    #vmax = np.quantile(flux[flux > 0.], 0.95)
    #scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
    #cmap = mpl.cm.get_cmap('binary_r').copy()
    cmap = mpl.colors.ListedColormap(['white']).copy()
    cmap.set_bad('black')
    
    #Plot Figure
    fontsize=20
    fig = plt.figure()
    ax.grid(False)
    im=ax.pcolormesh(time_dt64,frequency, v,cmap=cmap,shading='auto')
    
    #format axis 
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fontsize-6)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    #ax.set_title('True Mask', fontsize=fontsize + 2)
    dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(dateFmt)
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    if val == 's':
        cb.set_label(r'Flux Density'+'\n (W/m$^2$/Hz)', fontsize=fontsize-2)
    elif val =='v':
        cb.set_label('Normalized'+'\n Degree of'+'\n Circular Polarization', fontsize=fontsize-2)
        
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.utcfromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_,  color='orangered', linewidth=2, alpha=1, fill=False,label='Predicted Polygon'))
    legend_without_duplicate_labels(ax,'lower right')
    ax.set_xlim(time_dt64[0], time_dt64[-1])
    
    plt.close(fig)
    return ax


def plot_flux(ax,time_view_start, time_view_end, file, colour_in=None,
              frequency_lines=None, fontsize=20):
    
    #Load data from .sav file
    time, freq, flux = extract_data(file, time_view_start=time_view_start,\
                                    time_view_end=time_view_end,val='s')
    #Parameters for colorbar
    #This is the function that does flux normalisation based on s/c location
    vmin, vmax=plt_func.flux_norm(time[0], time[-1])   #change from log10 to actual values.
    #clrmap ='viridis'
    #vmin = np.quantile(flux[flux > 0.], 0.05)
    #vmax = np.quantile(flux[flux > 0.], 0.95)
    #vmin = 1e-24
    #vmax = 1e-20
    scaleZ = colors.LogNorm(vmin, vmax)
    cmap = mpl.cm.get_cmap("viridis").copy()
    #cmap.set_bad('black')
    scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
    
    #Make figure
    fig = plt.figure( edgecolor='white', frameon=False)
    im=ax.pcolormesh(time, freq, flux, norm=scaleZ,cmap=cmap,  shading='auto')
    ax.set_yscale('log')
    
    #format axis 
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    #ax.get_xaxis().set_visible(False)
    ax.tick_params(axis='y',labelsize=fontsize-5, labelcolor='k')
    ax.tick_params(axis='x', labelcolor='w')
    #ax.set_xticks(color='w')
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize, color='k')
    #ax.set_xlabel('Time', fontsize=fontsize)
    
    ######### X label formatting ###############
    
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    #normal
    #dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    #ax.xaxis.set_major_formatter(dateFmt)
    
    #For using trajectory data
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(plt_func.ephemeris_fmt_hour_tick))
    #eph_str = '\n'.join(['DOY\n',
     #           r'$R_{sc}$ ($R_{S}$)',
      #          r'$\lambda_{sc}$ ($^{\circ}$)',
       #         r'LT$_{sc}$ (Hrs)'])
    #kwargs = {'xycoords': 'figure fraction',
     #   'fontsize': fontsize-6}
    #kwargs['xy'] = (0.03, 0.009)
    #ax.annotate(eph_str,**kwargs)
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.set_label(r'Flux Density'+'\n (W/m$^2$/Hz)', fontsize=fontsize-2, color='k')
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    
    #For adding horizontal lines at specific frequencies
    if frequency_lines is not None:
        for i in frequency_lines:
            ax.hlines(i, time[0], time[-1], colors = 'darkslategray',linewidth=1,linestyles='--', label='{}kHz'.format(i))
          
     #For plotting polygons onto spectrogram.
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.utcfromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_, color='orangered', linewidth=2, alpha=1, fill=False,label='Predicted Polygon'))
    legend_without_duplicate_labels(ax,'upper right')
    ax.set_xlim(time[0], time[-1])
    plt.close(fig)
    return ax
def plot_test_res(ax, time_view_start, time_view_end, colour_in, fontsize=20):
    time = t_dt[(t_dt >= time_view_start) & (t_dt <= time_view_end)]
    s = test_results[:, (t_dt >= time_view_start) & (t_dt <= time_view_end)]
    frequency_tmp = np.load(output_data_fp + '/frequency.npy')
    step = (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/383
    frequency = np.flip(10**(np.arange(np.log10(frequency_tmp[0]),
                    np.log10(frequency_tmp[-1])+step/2, step, dtype=float)))
    
    fig = plt.figure(edgecolor='white',frameon=False)
    im = ax.pcolormesh(time, frequency, s, cmap='binary_r')
    ax.set_yscale('log')
    ax.tick_params(labelsize=fontsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax.xaxis.set_major_formatter(dateFmt)
    ax.tick_params(labelsize=fontsize)
    ax.set_title('Predicted Mask', fontsize=fontsize)
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.ax.tick_params(labelsize=fontsize-2)
    cb.set_label('Probability', fontsize=fontsize)
    for shape in colour_in:
        shape_=shape.copy()
        shape_[:,0]=[mdates.date2num(datetime.utcfromtimestamp(i)) for i in shape_[:,0]]
        ax.add_patch(Polygon(shape_, color='orangered', linewidth=2, alpha=1, fill=False, label='Predicted Polygon'))
    
    ax.set_xlim(time[0], time[-1])
    plt.close(fig)
    return ax
    
def plot_pol(ax,time_view_start, time_view_end, file,colour_in=None,frequency_lines=None, fontsize=20):
    
    #Load data from .sav file
    time, freq, pol = extract_data(file, time_view_start=time_view_start, \
                                   time_view_end=time_view_end,val='v')
    #Parameters for colorbar
    vmin=-1
    vmax=1
    clrmap ='binary'
    scaleZ = colors.Normalize(vmin=vmin, vmax=vmax)
    
    #Make figure
    fig = plt.figure(edgecolor='white',frameon=False)
    ax.grid(False)
    im=ax.pcolormesh(time, freq, pol, norm=scaleZ, cmap=clrmap, shading='auto')
    ax.set_yscale('log')
    
    
    #format axis 
    #ax.get_xaxis().set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-3, labelcolor='k')
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize, color='k')
    #Uncomment to set title
    #ax.set_title(f'{time_view_start} to {time_view_end}', fontsize=fontsize + 2)
    
    ######### X label formatting ###############
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    #normal
    dateFmt = mdates.DateFormatter('%d\n%H:%M')
    ax.xaxis.set_major_formatter(dateFmt)
    
    #For using trajectory data
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(plt_func.ephemeris_fmt_hour_tick))
    #eph_str = '\n'.join(['Time (UTC)\n',
     #          r'$R_{sc}$ ($R_{S}$)',
      #          r'$\lambda_{sc}$ ($^{\circ}$)',
       #         r'LT$_{sc}$ (Hrs)'])
    #kwargs = {'xycoords': 'figure fraction',
     #  'fontsize': fontsize-7}
    #kwargs['xy'] = (0.02, 0.045)
    #ax.annotate(eph_str,**kwargs)
    #ax.text(-0.16, -0.22, eph_str, horizontalalignment='center',verticalalignment='center',
     #       transform = ax.transAxes,fontsize=fontsize-5, weight=300)
    
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.set_label('Degree of'+'\n Circular Polarization', fontsize=fontsize-2,color='k')
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    
    #For adding horizontal lines at specific frequencies
    if frequency_lines is not None:
        for i in frequency_lines:
            ax.hlines(i, time[0], time[-1], colors = 'darkslategray',linewidth=1,linestyles='--', label='{}kHz'.format(i))
              
    #For plotting polygons onto spectrogram.
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.utcfromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_, color='orangered', linewidth=2, alpha=1, fill=False, label='Predicted Polygon'))
    
    ax.set_xlim(time[0], time[-1])
    plt.close(fig)
    return ax

def calc_dur(l1, l2):
    return (l2-l1).total_seconds()/3600
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

def make_dataframe(file):
    #Timestamps is in the form of pandas timestamp, but you can edit the lfe_coordinates function
    #if you would like it in a different format.
    timestamps, freqs, feature, id_ = lfe_coordinates(file)
    
    
    #Start and end times of each labelled item.
    start = [min(i) for i in timestamps]
    end = [max(i) for i in timestamps]
    f_min=[min(i) for i in freqs]
    f_max=[max(i) for i in freqs]
    df = pd.DataFrame({'start': start, 'end':end,'label':feature,'id':id_, 'f_min':f_min,'f_max':f_max})
    df = df.drop_duplicates(subset='start', keep='first')
    df=df.sort_values(by='start').reset_index(drop=True)
    df['dur']=df.apply(lambda x: (x.end-x.start).total_seconds()/3600, axis=1)

    return df
def freq_evolution(freq, time):
    min_t = min(time)
    min_t_ind = list(time).index(min_t)
    min_f = freq[min_t_ind]
    max_t = max(time)
    max_t_ind = list(time).index(max_t)
    max_f = freq[max_t_ind]
    before_f = freq[min_t_ind:max_t_ind+1]
    before_t = time[min_t_ind:max_t_ind+1]
    return before_f, before_t
def plot_freq_evolution(ax, freq, time):
    f, t = freq_evolution(freq, time)
    ax.tick_params(labelsize=17)
    t=[datetime.utcfromtimestamp(i) for i in t]
    ax.set_xlim(min(t)-pd.Timedelta(30,'minutes'), max(t)+pd.Timedelta(30,'minutes'))
    ax.hlines(80, min(t)-pd.Timedelta(30,'minutes'), max(t)+pd.Timedelta(30,'minutes'),color='gray',linestyle='dashed')
    ax.set_ylabel('$\Delta$f (kHz)',fontsize=20)
    #ax.set_xlabel('Time')
    #ax.set_title(f'Frequency Evolution of single LFE, duration = {df.dur[i]} hrs')
    ax.plot(t,f, linewidth=0.5)
    ax.get_xaxis().set_visible(False)
   # ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.set_yscale('log')
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cax.set_facecolor('none')
    for axis in ['top','bottom','left','right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])
    return ax


polygon_fp = output_data_fp + f"/{model_name}/test_2006_selected_contours.json"
plt.ioff()
starts = [[pd.Timestamp('20060228'), pd.Timestamp('20061027')],[pd.Timestamp('20060714T06:00'),
                                            pd.Timestamp('20060214')]]
ends =[[pd.Timestamp('20060302'), pd.Timestamp('20061028T10:00')], [pd.Timestamp('20060714T19:00'),
                                    pd.Timestamp('20060215T03:00')]]

plt.ioff()
panel_labels = ['a', 'b', 'c', 'd']
count=0
for pan in range(2):
    os.system('cls')
    plt.ioff()
    fig, ax = plt.subplots(3, 2, figsize=(25, 20))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    starts_=starts[pan]
    ends_ = ends[pan]
    for num in range(2):
        s = starts_[num]
        e = ends_[num]
        [ax1, ax2, ax3] = ax[:, num]
        year = datetime.strftime(s, '%Y')
        if year == '2017':
            file = input_data_fp + '/SKR_2017_001-258_CJ.sav'
        else:
            file = input_data_fp + f'/SKR_{year}_CJ.sav'
        colour_in = get_polygons(polygon_fp, s, e)

        ax1 = plot_flux(ax1, s, e, file,  colour_in)
        ax1.text(-0.075,1.05, panel_labels[count], horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes,fontsize=30, weight='bold')
        ax2 = plot_pol(ax2, s, e, file,  colour_in)
        ax3 = plot_mask(ax3, s, e, 's', file, 'C:/Users/eliza/Desktop/Python_Scripts/input_data/SKR_LFEs.json',colour_in)
        print(count)
        count+=1
    s_str=datetime.strftime(starts_[0], '%Y-%m-%dT%H')
    e_str = datetime.strftime(ends_[-1], '%Y-%m-%dT%H')
    fp_save = output_data_fp + f'/{model_name}/figures/2006_prediction_{s_str}-{e_str}.png'
    plt.savefig(fp_save, bbox_inches='tight')
    plt.close(fig)