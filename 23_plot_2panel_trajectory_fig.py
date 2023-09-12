# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:31:25 2023

@author: eliza
"""

from matplotlib.ticker import (MultipleLocator)
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import date
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'
plt.style.use('default')
def roundup(x):
     return x if x % 100 == 0 else x + 100 - x % 100
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),loc='lower right', fontsize=16)
def datetime_todoy2004(x):
    year = int(datetime.strftime(x,'%Y'))
    f_date = date(2004, 1, 1)
    l_date = date(year, 1, 1)
    delta = l_date - f_date
    #Calculates the number of days since day 1 of 1997 to day 1 of given year.
    days=delta.days
    
    doy_current= ((pd.to_datetime(x) -datetime(int(datetime.strftime(x,'%Y')),1,1)).total_seconds()/86400) +1
    
    doy_from04 = doy_current+days
    return doy_from04


df=pd.read_csv(input_data_fp + '/total_timestamps.csv', parse_dates=['start','end'])
LFE_labels = ['LFE','LFE_dg','LFE_ext','LFE_m','LFE_sp']
nolfe = df.loc[(df['label']=='NoLFE'),:].reset_index(drop=True)
lfe = df.loc[(df['label'].isin(LFE_labels)),:].reset_index(drop=True)
traj_data = pd.read_csv(input_data_fp + '/traj_df_allyears.csv',parse_dates=['datetime_ut'])


lfe_times=[]
for i,j in zip(lfe['start'], lfe['end']):
    a=traj_data.loc[traj_data["datetime_ut"].between(i,j),:]
    lfe_times.append(a)
lfe_times=pd.concat(lfe_times, axis=0)

nolfe_times=[]
for i,j in zip(nolfe['start'], nolfe['end']):
    a=traj_data.loc[traj_data["datetime_ut"].between(i,j),:]
    nolfe_times.append(a)
nolfe_times=pd.concat(nolfe_times, axis=0)


lfe_and_nolfe = pd.concat([lfe_times, nolfe_times], axis=0)
lfe_and_nolfe=lfe_and_nolfe.sort_values(by=['datetime_ut'])

latitude=traj_data['lat_krtp']
time = traj_data['doyfrac']
time_doy04 = traj_data['datetime_ut'].apply(datetime_todoy2004)

lfestart_doy=lfe['start'].apply(datetime_todoy2004)
nolfestart_doy=nolfe['start'].apply(datetime_todoy2004)

years=np.arange(2004, 2018, 1)
year_labels=[str(i) for i in years]
years_isot=[str(i)+'0101' for i in years]
years_datetime = [pd.Timestamp(i) for i in years_isot]
axis_labels=[datetime_todoy2004(i) for i in years_datetime]


#Define fig
plt.ioff()
fig, axes = plt.subplots(2,1,figsize=(14,10))
plt.subplots_adjust(hspace=1, wspace=0.5)
#plt.tick_params(axis='both', which='major', labelsize=18)
#fig.set_size_inches(1500, 1000)
ax=axes[0]
ax.text(-0.075,1.05, 'a', horizontalalignment='center',verticalalignment='center',transform = ax.transAxes,fontsize=20, weight='bold')
#ax.tick_params(axis='both', which='major', labelsize=18)
#set limites on axis
max_ = max(time_doy04)
ax.set_xlim(min(time_doy04), max(time_doy04))
max_=roundup(max(time_doy04))
bins=np.arange(0, max_+1, 50)
h, b = np.histogram(lfestart_doy, bins=bins)
ax.hist(b[:-1], b, weights=h, linewidth=2,color='orange',label='LFE',histtype='step')
h, b = np.histogram(nolfestart_doy, bins=bins)
ax.hist(b[:-1], b, weights=h, linewidth=2,color='skyblue',label='Non-LFE',histtype='step')
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_ylim(0, 165)
ax2 = ax.twinx()
ax.tick_params(axis='both', which='major', labelsize=18)
ax2.set_ylabel('Latitude ($^{\circ}$)',fontsize=18)
ax2.set_ylim(-90,90)
ax2.yaxis.set_major_locator(MultipleLocator(30))
ax2.yaxis.set_major_formatter('{x:.0f}')
ax2.tick_params(axis='both', which='major', labelsize=18)
# For the minor ticks, use no labels; default NullFormatter.
ax2.yaxis.set_minor_locator(MultipleLocator(10))
#ax2.set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax2.plot(time_doy04, latitude, color='gray',alpha=0.2)
ax.set_xlabel('DOY 2004',fontsize=22)
ax.set_ylabel('Occurence',fontsize=22)

ax4=ax.twiny()
ax4.tick_params(axis='both', which='major', labelsize=18)
ax4.set_xlim(min(time_doy04), max(time_doy04))
ax4.set_xticks(axis_labels,labels=year_labels)
ax.legend(fontsize=16)


#Panel 2
ax3 = axes[1]
ax3.text(-0.075,1.05, 'b', horizontalalignment='center',verticalalignment='center',transform = ax3.transAxes,fontsize=20, weight='bold')
ax3.tick_params(axis='both', which='major', labelsize=18)

splits = np.load(input_data_fp + '/orbit_indices.npy', allow_pickle=True)
x_ = np.split(traj_data['localtime'], splits)
y_=np.split(latitude, splits)
for i in range(len(x_)):
    ax3.plot(x_[i], y_[i],color='gray')  
for i, j in zip(lfe['start'], lfe['end']):
    traj = traj_data.loc[traj_data['datetime_ut'].between(i, j),:]
    ax3.plot(traj['localtime'], traj['lat_krtp'], linewidth=1,color='orange', label='LFE')
for i, j in zip(nolfe['start'], nolfe['end']):
    traj = traj_data.loc[traj_data['datetime_ut'].between(i, j),:]
    ax3.plot(traj['localtime'], traj['lat_krtp'],linewidth=1, color='skyblue', label='Non-LFE')
ax3.set_xlabel('Local Time (Hrs)',fontsize=22)
ax3.set_ylabel('Latitude ($^{\circ}$)',fontsize=22)
ax3.set_ylim(-90,90)
#plt.plot(time_doy04, latitude, color='gray',alpha=0.4)
ax3.yaxis.set_major_locator(MultipleLocator(30))
ax3.yaxis.set_major_formatter('{x:.0f}')
# For the minor ticks, use no labels; default NullFormatter.
ax3.yaxis.set_minor_locator(MultipleLocator(10))
ax3.set_xlim(0,24)
legend_without_duplicate_labels(ax3)
plt.tight_layout()
#plt.show()
plt.savefig(figure_fp + "/traj_fig_Smallbin.png")




