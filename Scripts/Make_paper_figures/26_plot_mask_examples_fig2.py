# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:26:21 2023

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
import shapely.geometry as sg
from os import path
from tfcat import TFCat
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'

def legend_without_duplicate_labels(ax, loc_):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc_,fontsize=14)


def get_polygons(polygon_fp,start, end):
    unix_start=t.mktime(start.utctimetuple())
    unix_end=t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array=[]
    if path.exists(polygon_fp):
        catalogue = TFCat.from_file(polygon_fp)
        for i in range(len(catalogue)):
                time_points=np.array(catalogue._data.features[i]['geometry']['coordinates'][0])[:,0]
                if any(time_points <= unix_end) and any(time_points >= unix_start):
                    polygon_array.append(np.array(catalogue._data.features[i]['geometry']['coordinates'][0]))
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
def load_ephem(dtime,dtime2):
    year=pd.to_datetime(dtime).year
    orbit_df = pd.read_csv(input_data_fp + '/trajectory{}.csv'.format(year), parse_dates=['datetime_ut'])
    orbit_df = orbit_df.loc[orbit_df['datetime_ut'].between(dtime,dtime2), :]
    return orbit_df
def flux_norm(dtime,dtime2):
    step=1
    bins = np.arange(0,24.1,step)
    decimals=0
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
            flux_arr = np.load(input_data_fp + '/lowlat_flux.npy', allow_pickle=True)
        else:
            flux_arr= np.load(input_data_fp + '/highlat_flux.npy', allow_pickle=True)
        
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
        flux_arr1 = np.load(output_data_fp + '/lowlat_flux.npy', allow_pickle=True)
        flux_arr1 = np.concatenate(flux_arr1, axis=0)
        flux_arr2= np.load(output_data_fp + '/highlat_flux.npy', allow_pickle=True)
        flux_arr2 = np.concatenate(flux_arr2, axis=0)
        ttl_flux_arr = np.concatenate([flux_arr1, flux_arr2], axis=0)
    p80 = np.percentile(ttl_flux_arr, 80)
    p10 = np.percentile(ttl_flux_arr, 10)
    return p10, p80

def plot_flux(ax,time_view_start, time_view_end, file, colour_in=None, frequency_lines=None, fontsize=25):
    
    #Load data from .sav file
    time, freq, flux = extract_data(file, time_view_start=time_view_start,\
                                    time_view_end=time_view_end,val='s')
    #Parameters for colorbar
    #This is the function that does flux normalisation based on s/c location
    vmin, vmax= flux_norm(time[0], time[-1]) 
    clrmap ='viridis'
    scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
    
    #Make figure
    fig = plt.figure( edgecolor='white', frameon=False)
    im=ax.pcolormesh(time, freq, flux, norm=scaleZ,cmap=clrmap,  shading='auto')
    ax.set_yscale('log')
    
    
    #format axis 
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-5)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)

    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.set_label(r'Flux Density'+'\n (W/m$^2$/Hz)', fontsize=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)

    #For adding horizontal lines at specific frequencies
    if frequency_lines is not None:
        for i in frequency_lines:
            ax.hlines(i, time[0], time[-1], colors = 'darkslategray',linewidth=1,linestyles='--', label='{}kHz'.format(i))
          
     #For plotting polygons onto spectrogram.
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.utcfromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_, color='black', linestyle='dashed',linewidth=4, alpha=1, fill=False,label='Predicted Polygon'))
    #legend_without_duplicate_labels(ax,'lower right')
    ax.set_xlim(time[0], time[-1])
    plt.close(fig)
    return ax
def ephemeris_labels(dtime):
    

    year=pd.to_datetime(dtime).year
    fp=input_data_fp + '/trajectory{}.csv'.format(year)
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
    tick_str = datetime.strftime(tick_dt, ('%Y-%m-%d %H:%M'))
    tick_str = tick_str.replace(' ', '\n')
    # this returns corresponding radial dist, gse_lat, gse_lt for the tick
    # as strings in a list
    eph_str = ephemeris_labels(tick_dt)
    eph_str = [tick_str] + eph_str
    tick_str = '\n'.join(eph_str)

    return tick_str
def plot_mask(ax,time_view_start, time_view_end, val,
              file_data,polygon_fp,colour_in=None, fontsize=25):
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
    cmap = colors.ListedColormap(['white']).copy()
    cmap.set_bad('black')
    
    #Plot Figure
    fig = plt.figure()
    im=ax.pcolormesh(time_dt64,frequency, v,cmap=cmap,shading='auto')
    
    #format axis 
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fontsize-5)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    #ax.set_title('True Mask', fontsize=fontsize + 2)
    #dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    #ax.xaxis.set_major_formatter(dateFmt)

    #For using trajectory data
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(ephemeris_fmt_hour_tick))
    eph_str = '\n'.join(['Time (UTC)\n',
                r'$R_{sc}$ ($R_{S}$)',
                r'$\lambda_{sc}$ ($^{\circ}$)',
                r'LT$_{sc}$ (Hrs)'])
    ax.text(-0.16, -0.210, eph_str, horizontalalignment='center',verticalalignment='center',transform = ax.transAxes,fontsize=fontsize-6, weight=300)
    
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cax.set_facecolor('none')
    for axis in ['top','bottom','left','right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])
    #cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    #if val == 's':
     #   cb.set_label(r'Flux Density'+'\n (W/m$^2$/Hz)', fontsize=fontsize-2)
    #elif val =='v':
     #  cb.set_label('Normalized'+'\n Degree of'+'\n Circular Polarization', fontsize=fontsize-2)
        
    #cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.utcfromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_,  color=(0.163625, 0.471133, 0.558148), linestyle='dashed',linewidth=4, alpha=1, fill=False,label='Predicted Polygon'))
    ax.set_xlim(time_dt64[0], time_dt64[-1])
    #legend_without_duplicate_labels(ax,'lower right')
    plt.close(fig)
    return ax
def plot_pol(ax,time_view_start, time_view_end, file,colour_in=None,frequency_lines=None, fontsize=25):
    
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
    im=ax.pcolormesh(time, freq, pol, norm=scaleZ, cmap=clrmap, shading='auto')
    ax.set_yscale('log')
    
    
    #format axis 
    ax.tick_params(axis='both', which='major', labelsize=fontsize-3)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    
    ######### X label formatting ###############
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.get_xaxis().set_visible(False)
    #For using trajectory data
    #ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(ephemeris_fmt_hour_tick))
    #eph_str = '\n'.join(['Time (UTC)\n',
     #           r'$R_{sc}$ ($R_{S}$)',
      #          r'$\lambda_{sc}$ ($^{\circ}$)',
       #         r'LT$_{sc}$ (Hrs)'])
    #kwargs = {'xycoords': 'figure fraction',
     # 'fontsize': fontsize-7}
    #kwargs['xy'] = (0.02, 0.045)
    #ax.annotate(eph_str,**kwargs)
    #ax.text(-0.16, -0.210, eph_str, horizontalalignment='center',verticalalignment='center',transform = ax.transAxes,fontsize=fontsize-6, weight=300)
    
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.set_label('Degree of'+'\n Circular Polarization', fontsize=fontsize-2)
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
            ax.add_patch(Polygon(shape_, color=(0.163625, 0.471133, 0.558148), linestyle='dashed',linewidth=4, alpha=1, fill=False, label='Predicted Polygon'))
    #legend_without_duplicate_labels(ax,'lower right')
    ax.set_xlim(time[0], time[-1])
    plt.close(fig)
    return ax


polygon_fp = input_data_fp + '/SKR_LFEs.json'
total_df=pd.read_csv(input_data_fp + "/total_timestamps.csv", parse_dates=['start','end'])
total_df_nosm = total_df.loc[total_df['label']!='LFE_sm', :].reset_index(drop=True)

plt.ioff()
inds = [222, 1447]
#col_1 = [222, 1447]
#col_2 = [325,719]
#col_3 = [148, 567]
#cols = [col_1, col_2, col_3]
lab1 = ['LFE', 'NoLFE']
panel_labels = [['a', 'b', 'c'],['d','e','f']]
#lab2 = [r'LFE$_{sp}$',r'LFE$_{sm}$']
#lab3 = [r'LFE$_{dg}$',  r'LFE$_{ext}$']
#labs = [lab1, lab2, lab3]
#panel_labels = [['a','b','c'], ['d','e','f']]
#panels = [panel_1, panel_2, panel_3]
plt.ioff()
fig, axes = plt.subplots(3, 2, figsize=(34, 18))
plt.subplots_adjust(hspace=0.1, wspace=0.5)
for num in range(2):
    i = inds[num]
    ax1 = axes[0, num]
    ax2 = axes[1, num]
    ax3 = axes[2, num]
    label_ = lab1[num]
    panel_label = panel_labels[num]
    start = total_df_nosm.loc[i, 'start']
    end = total_df_nosm.loc[i, 'end']
    year = datetime.strftime(start, '%Y')
    if year == '2017':
        file = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else:
        file = input_data_fp + f'/SKR_{year}_CJ.sav'
    colour_in = get_polygons(polygon_fp, start, end)
    ax1 = plot_flux(ax1, start, end, file,  colour_in, frequency_lines=[10, 40, 100, 400])
    ax1.text(-0.075,1.05, panel_label[0], horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes,fontsize=30, weight='bold')
    ax1.set_title(f'{label_} Class', fontsize=30, pad=15)
    ax2.text(-0.075,1.05, panel_label[1], horizontalalignment='center',verticalalignment='center',transform = ax2.transAxes,fontsize=30, weight='bold')
    ax2 = plot_pol(ax2, start, end, file,  colour_in, frequency_lines=[10, 40, 100, 400])
    ax3 = plot_mask(ax3, start, end, 's',file,  polygon_fp)
    ax3.text(-0.075,1.05, panel_label[2], horizontalalignment='center',verticalalignment='center',transform = ax3.transAxes,fontsize=30, weight='bold')
fp_save = figure_fp + '/mask_example_plot.png'
plt.savefig(fp_save, bbox_inches='tight')
plt.close(fig)


