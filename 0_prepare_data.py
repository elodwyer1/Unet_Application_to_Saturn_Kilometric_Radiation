# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 19:05:27 2022

@author: eliza
"""
import pandas as pd
from datetime import datetime
import configparser
config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']
def traject(year):
    #Filepath to trajectory data in KRTP coordinate system.
    fp_krtp = input_data_fp + "/{}_FGM_KRTP_1M.TAB".format(year)
    #Read in trajectory data.
    df_krtp = pd.read_csv(fp_krtp, header=None,delim_whitespace=True)
    #Rename columns to informative column names.
    df_krtp = df_krtp.rename(columns={0: 'Time_isot',1:'B_r(nT)',2:'B_theta(nT)',3:'B_phi(nT)',
                            4:'B_total(nT)',5:'Range(R_s)',6:'Latitude',7: "East_Longitude",
                            8:'Local_Time',9:'NPTS'})
    #Function that datetime timestamp to day of year (DOY) of given year.
    func_tstmp_todoy = lambda x: ((x -datetime(int(datetime.strftime(x,'%Y')),1,1)).total_seconds()/86400) +1
    #Convert datetime string (isoformat) to datetime timestamp.
    time_dt = df_krtp['Time_isot'].apply(lambda x: datetime.fromisoformat(x))
    #Converted to DOY fraction.
    doy_frac = list(map(func_tstmp_todoy, time_dt))
    #Filepath to trajectory data in KSM coordinate system.
    fp_ksm = input_data_fp + "/{}_FGM_KSM_1M.TAB".format(year)
    #Read in traj. file.
    df_ksm=pd.read_csv(fp_ksm, header=None,delim_whitespace=True)
    #Rename columns to informative column names.
    df_ksm = df_ksm.rename(columns={0: 'Time_isot',1:'B_x(nT)',2:'B_y(nT)',3:'B_z(nT)',
                            4:'B_total(nT)',5:'X(R_s)',6:'Y(R_s)',7: "Z(R_s)",
                            8:'Local_Time',9:'NPTS'})
    #Combine trajectory dataframes in different coordinate systems to one, with magnetic field data included.
    df_final = pd.DataFrame({'datetime_ut': time_dt, 'bphi_krtp':df_krtp['B_phi(nT)'],
                                    'br_krtp':df_krtp['B_r(nT)'],
                                    'btheta_krtp':df_krtp['B_theta(nT)'],
                                    'btotal':df_krtp['B_total(nT)'],
                                    'doyfrac':doy_frac,
                                    'lat_krtp':df_krtp['Latitude'],
                                    'localtime':df_krtp['Local_Time'],
                                    'range':df_krtp['Range(R_s)'],
                                    'xpos_ksm':df_ksm['X(R_s)'],
                                    'ypos_ksm':df_ksm['Y(R_s)'],
                                    'zpos_ksm':df_ksm['Z(R_s)']})
    return df_final
total_traj = []
for year in range(2004,2018, 1):
    print(year)
    #Dataframe with trajectory information.
    #columns are: 'datetime_ut','bphi_krtp' 'br_krtp' 'btheta_krtp' 'btotal' 'doyfrac'
     # 'lat' 'localtime'  'range' 'xpos_ksm' 'ypos_ksm' 'zpos_ksm'
    traj_df = traject(year)  
    fp = input_data_fp + '/trajectory{}.csv'.format(year)
    traj_df.to_csv(fp, index=False)
    total_traj.append(traj_df)
total_traj = pd.concat(total_traj)
total_traj.to_csv(input_data_fp +'/traj_df_allyears.csv',index=False)

