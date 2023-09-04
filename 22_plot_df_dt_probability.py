# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 09:39:35 2023

@author: eliza
"""

from mpl_toolkits import axes_grid1
from astropy.time import Time
from tfcat import TFCat
import matplotlib.pyplot as plt
import pandas as pd
import configparser
from datetime import datetime
import numpy as np
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']

def get_data(file):
    co = []
    id_ = []
    feature=[]
    catalogue = TFCat.from_file(file)
    for i in range(len(catalogue._data['features'])):
        label=catalogue._data['features'][i]['properties']['feature_type']
        feature.append(label)
        id_.append(catalogue._data['features'][i]['id'])
        coords=np.array(catalogue._data['features'][i]['geometry']['coordinates'][0])
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

def make_dataframe(timestamps, freqs):
    #Timestamps is in the form of pandas timestamp, but you can edit the lfe_coordinates function
    #if you would like it in a different format.
    #Start and end times of each labelled item.
    start = [min(i) for i in timestamps]
    end = [max(i) for i in timestamps]
    del_f = [(max(i) - min(i)) for i in freqs]
    
    
    del_t = [(j-i).total_seconds()/3600 for i, j in zip(start, end)]
    df = pd.DataFrame({'start': start, 'end':end,'del_t':del_t, 'del_f':del_f})
    return df

## plot probability vs frequency/time 
data_name = '2004001_2017258'
plt.style.use('ggplot')
test_selected = pd.read_csv(output_data_fp + f'/{model_name}/{data_name}_selected_contours.csv')
cata = pd.read_csv(output_data_fp + f'/{model_name}/{data_name}_catalogue.csv', 
                   parse_dates=['start', 'end'])
calc_dur = lambda start, end: ((end-start).total_seconds())/3600
dur =  cata.apply(lambda x: calc_dur(x.start, x.end), axis=1)

plt.tick_params(labelsize=15)
plt.ioff()
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
test_selected['t_hours'] = (test_selected['delta_t'].reset_index(drop=True)*3)/60
im = ax.scatter(dur, test_selected['delta_f_khz'], s=8,
            c=test_selected['probability'], cmap='Blues')
ax.set_xscale('log')
ax.set_xlabel('$\Delta$t (hours)', fontsize=12)
ax.set_ylabel('$\Delta$f (kHz)', fontsize=12)
cbar = fig.colorbar(im, label='Probability')
plt.savefig(output_data_fp + f'/{model_name}/figures/{data_name}_probability_dt_df.png')
plt.clf()

fp_sav = input_data_fp + '/SKR_LFEs.json'
t, f, ft, id_ = lfe_coordinates(fp_sav)
df = make_dataframe(t, f)

f = 20
plt.ioff()
fig, ax = plt.subplots(1,2, figsize=(23, 8))
#Training data
ax[0].text(-0.075,1.05, 'a', horizontalalignment='center',verticalalignment='center',transform = ax[0].transAxes,fontsize=f+6, weight='bold')
ax[1].text(-0.075,1.05, 'b', horizontalalignment='center',verticalalignment='center',transform = ax[1].transAxes,fontsize=f+6, weight='bold')
ax[0].tick_params(labelsize=f)
ax[1].tick_params(labelsize=f)
im = ax[0].scatter(df['del_t'], df['del_f'], s=8,c=np.repeat(0.5, 984),
                cmap='Blues_r')
ax[0].set_xlim(0.2, 200)
ax[0].set_xscale('log')
ax[0].set_xlabel('$\Delta$t (hours)', fontsize=f+4)
ax[0].set_ylabel('$\Delta$f (kHz)', fontsize=f+4)
ax[0].set_title('Training LFEs', fontsize=f+6)
# create empty space to fit colorbar
divider = axes_grid1.make_axes_locatable(ax[0])
cax = divider.append_axes("right", size=0.15, pad=0.2)
cax.set_facecolor('none')
for axis in ['top','bottom','left','right']:
    cax.spines[axis].set_linewidth(0)
cax.set_xticks([])
cax.set_yticks([])

#Predicted data
im2 = ax[1].scatter(dur, test_selected['delta_f_khz'], s=8,
            c=test_selected['probability'], cmap='Blues')
ax[1].set_xlim(0.2, 200)
ax[1].set_xscale('log')
ax[1].set_xlabel('$\Delta$t (hours)', fontsize=f+4)
ax[1].set_ylabel('$\Delta$f (kHz)', fontsize=f+4)
ax[1].set_title('Predicted LFEs',fontsize=f+6)
divider = axes_grid1.make_axes_locatable(ax[1])
cax = divider.append_axes("right", size=0.15, pad=0.2)
cb = fig.colorbar(im2, extend='both', shrink=0.9, cax=cax, ax=ax[1])
cb.set_label('Probability', fontsize=f)
cb.ax.tick_params(labelsize=f-2)
#save figure
plt.tight_layout()
plt.savefig(output_data_fp + f'/{model_name}/figures/train_predicted_probability_dt_df.png')
plt.clf()