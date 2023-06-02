# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:21:44 2023

@author: eliza
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.ticker import MultipleLocator
import configparser
import os
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data'] 
model_name = config['filepaths']['model_name']
figure_fp = f'{output_data_fp}/{model_name}/figures'
if not os.path.exists(figure_fp):
    os.makedirs(figure_fp)
def twodhist(df):
    scatter_x = np.array(df['delta_f'])
    scatter_y = np.array(df['delta_t'])
    xedges=np.arange(0, 390, 5)
    yedges = np.arange(0, 135, 5)

    H, x, y = np.histogram2d(scatter_x, scatter_y, bins=(xedges, yedges))
    H=(H).T
    return H


#load in dataframe contain features pertaining to contours from model results.
test_df = pd.read_csv(output_data_fp + f'/{model_name}' + '/contour_analysis_test_data.csv')
train_df = pd.read_csv(output_data_fp + f'/{model_name}' + '/contour_analysis_train_data.csv')
plt.style.use('seaborn')
##### Plot Aspects of Contours
######### DELTA F
#Train
bins=np.arange(0, 400, 10)
plt.ioff()
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 6))
plt.rcParams.update({'font.size': 14})
ax1.tick_params(labelsize=12)
ax1.hist(train_df['delta_f'], bins)
ax1.set_xlim(0, 384)
ax1.set_xlabel('Delta f (pixels)')
ax1.set_ylabel('Counts')
ax1.set_title('Train Set')
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.tick_params('both', length=6, width=1, which='major')
ax1.tick_params('both', length=1.5, width=1, which='minor')
ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_major_formatter('{x:.0f}')
#Test
ax2.tick_params(labelsize=12)
ax2.hist(test_df['delta_f'], bins)
ax2.set_xlim(0, 384)
ax2.set_xlabel('Delta f (pixels)')
ax2.set_ylabel('Counts')
ax2.set_title('Test Set')
ax2.xaxis.set_minor_locator(MultipleLocator(10))
ax2.tick_params('both', length=6, width=1, which='major')
ax2.tick_params('both', length=1.5, width=1, which='minor')
ax2.xaxis.set_major_locator(MultipleLocator(50))
ax2.xaxis.set_major_formatter('{x:.0f}')
plt.tight_layout()
plt.savefig(figure_fp + '/hist_polydeltaf_trainandtest.png')
plt.clf()
###### 2D HISTOGRAM
#make figure
plt.ioff()
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(15, 9))
##Panel 1 Train Data: 2D histograms of delta t vs delta f for LFE and non-LFE data
plt.rcParams.update({'font.size': 14})
#ax1.hlines(10, 0, 384, ls='dotted', color='gray')
ax1.vlines(100, 0, 128, ls='dotted', color='gray')
#ax2.hlines(10, 0, 384, ls='dotted', color='gray')
ax2.vlines(100, 0, 128, ls='dotted', color='gray')
ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
ax1.set_xlim(0, 384)
ax2.set_ylim(0, 128)
ax1.set_xlim(0, 384)
ax2.set_ylim(0, 128)
#make histograms
H1=twodhist(test_df.loc[test_df['min_f']<=100, ])
H2=twodhist(test_df.loc[test_df['min_f']>100, ])
H3=twodhist(test_df.loc[test_df['label']!='NoLFE', ])
H4=twodhist(test_df.loc[test_df['label']=='NoLFE', ])
#colorbar normalisation
mins = [x[x != 0].min() for x in [H1, H2, H3, H4]]
maxs = [x[x != 0].max() for x in [H1, H2, H3, H4]]
vmin = min(mins)
vmax= max(maxs)
norm=matplotlib.colors.LogNorm(vmin, vmax)
xedges=np.arange(0, 390, 5)
yedges = np.arange(0, 135, 5)
X, Y = np.meshgrid(xedges, yedges)
#plot
im=ax1.pcolormesh(X, Y, H1, norm=norm,cmap='viridis')
im=ax2.pcolormesh(X, Y, H2, norm=norm,cmap='viridis')
#format axes
cb=fig.colorbar(im, ax=ax1)
cb.set_label('Number of counts')
cb=fig.colorbar(im, ax=ax2)
cb.set_label('Number of counts')
ax1.set_xlabel('Delta f (pixels)')
ax1.set_ylabel('Delta t (pixels)')
ax1.set_title('Test Set: minimum frequency >= 100kHz')
ax2.set_xlabel('Delta f (pixels)')
ax2.set_ylabel('Delta t (pixels)')
ax2.set_title('Test Set: minimum frequency < 100kHz')


##Panel 2 Test Data: 2D histograms of delta t vs delta f for LFE and non-LFE data
#format axes
ax3.vlines(100, 0, 128, ls='dotted', color='gray')
ax4.vlines(100, 0, 128, ls='dotted', color='gray')
ax3.tick_params(labelsize=12)
ax4.tick_params(labelsize=12)
ax3.set_xlim(0, 384)
ax4.set_ylim(0, 128)
ax3.set_xlim(0, 384)
ax4.set_ylim(0, 128)
ax3.set_xlabel('Delta f (pixels)')
ax3.set_ylabel('Delta t (pixels)')
ax3.set_title('Test Set: Images with LFE')
ax4.set_xlabel('Delta f (pixels)')
ax4.set_ylabel('Delta t (pixels)')
ax4.set_title('Test Set: Images without LFE')
#plot histogram
im=ax3.pcolormesh(X, Y, H3, norm=norm,cmap='viridis')
im=ax4.pcolormesh(X, Y, H4, norm=norm,cmap='viridis')
#plot colorbar
cb=fig.colorbar(im, ax=ax3)
cb.set_label('Number of counts')
cb=fig.colorbar(im, ax=ax4)
cb.set_label('Number of counts')

#Save figure
plt.tight_layout()
plt.savefig(figure_fp + '/2dhist_deltat_deltaf_trainandtest.png')
plt.clf()
