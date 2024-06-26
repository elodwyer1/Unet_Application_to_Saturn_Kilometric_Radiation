# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:46:12 2023

@author: eliza
"""

import numpy as np
from scipy.io import readsav
import configparser
import pandas as pd
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']

#each year of the mission
years=np.arange(2004, 2018, 1)
#empty lists for appending time, flux and polarization values.
t=[] #time
s=[] #flux
v=[] #polarization
#zip through each year and append time, flux and pol. data to corresponding list.
for year in years:
    if year == 2017:
        file = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        file = input_data_fp + '/SKR_{}_CJ.sav'.format(year)
    data = readsav(file, python_dict=True)
    t.append(data['t'])
    s.append(data['s'])
    v.append(data['v'])
#join each list into one larger list.     
all_t = np.concatenate(t)
all_s = np.concatenate(s, axis=1)
all_v = np.concatenate(v, axis=1)
f=np.array(data['f'])  
doy_one = pd.Timestamp(str(1997)) - pd.Timedelta(1, 'D')
t_timestamp = np.array([doy_one + pd.Timedelta(t * 1440, 'm') for t in all_t],
    dtype=pd.Timestamp)
#save lists to input data filepath.
s = np.save(input_data_fp + "/s_all_years.npy", all_s)
v = np.save(input_data_fp + "/p_all_years.npy",all_v)
t_doy=np.save(input_data_fp + "/time_all_years.npy", all_t)
np.save(input_data_fp + "time_indatetime_all_years.npy", t_timestamp)

