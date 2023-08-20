# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 09:39:35 2023

@author: eliza
"""

import json
from datetime import datetime
from shapely.validation import make_valid
import cv2
from shapely.geometry import LinearRing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from tfcat import TFCat
from shapely.geometry import Point, Polygon, LineString, MultiPoint
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
model_name = config['filepaths']['model_name']



## plot probability vs frequency/time 
data_name = '2004001_2017258'
plt.style.use('ggplot')
test_selected = pd.read_csv(output_data_fp + f'/{model_name}/test_{data_name}_selected_contours.csv')


plt.tick_params(labelsize=15)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
#ax.set_title('LFEs detected in 2006 using input method with overlapping time ranges', 
 #            fontsize=12)

t_hours = (test_selected['delta_t'].reset_index(drop=True) * 3)/60
im = ax.scatter(t_hours, test_selected['delta_f_khz'], s=8,
            c=test_selected['probability'], cmap='Blues')
ax.set_xscale('log')
ax.set_xlabel('$\Delta$t (hours)', fontsize=12)
ax.set_ylabel('$\Delta$f (kHz)', fontsize=12)
cbar = fig.colorbar(im, label='Probability')
plt.show()
plt.savefig(output_data_fp + f'/{model_name}/figures/{data_name}_probability_dt_df.png')


