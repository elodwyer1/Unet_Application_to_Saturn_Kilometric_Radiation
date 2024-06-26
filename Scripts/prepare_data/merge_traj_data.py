# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sqlite3
import logging
from ..read_config import config

def traject(year: int, config: config):
    # Filepaths
    fp_krtp = f"{config.input_data_fp}/{year}_FGM_KRTP_1M.csv"
    fp_ksm = f"{config.input_data_fp}/{year}_FGM_KSM_1M.csv"
    #, names=['TIME', 'BR', 'BTHETA', 'BPHI', 'BTOTAL', 'RANGE', 'LATITUDE', 'EAST LONGITUDE', 'LOCAL HOUR', 'NPTS']
    #, names=['TIME', 'BX', 'BY', 'BZ', 'BTOTAL', 'X', 'Y', 'Z', 'LOCAL HOUR', 'NPTS']
    # Read trajectory data and rename columns
    df_krtp = pd.read_csv(fp_krtp)
    df_ksm = pd.read_csv(fp_ksm)

    df_krtp = df_krtp[['TIME', 'RANGE', 'LATITUDE', 'LOCAL HOUR']]
    df_ksm = df_ksm[['TIME', 'X', 'Y', 'Z']]

    # Convert datetime string to datetime object and calculate DOY fraction
    df_krtp['datetime_ut'] = pd.to_datetime(df_krtp['TIME'], format='mixed')
    df_krtp['doyfrac'] = [(time - datetime(time.year, 1, 1)).total_seconds() / 86400 + 1 for time in df_krtp['datetime_ut']]

    # Combine dataframes
    df_final = pd.concat([df_krtp[['datetime_ut', 'doyfrac', 'LATITUDE', 'LOCAL HOUR', 'RANGE']], df_ksm[['X', 'Y', 'Z']]], axis=1)

    return df_final

def save_to_sqldb(df: pd.DataFrame, config: config):
    conn = sqlite3.connect(f"{config.proc_data_fp}/segment_radio.db")
    df.to_sql(f"traj_df_allyears", conn, if_exists='replace', index=False)
    # Close the connection
    conn.close()

def process_traj(config: config):
    print('Processing trajectory data')
    # Process trajectory data for each year
    total_traj = []
    for year in tqdm(range(2004, 2018)):
        total_traj.append(traject(year, config))
    # Concatenate dataframes for all years and save to CSV
    total_traj = pd.concat(total_traj).reset_index(drop=True)
    save_to_sqldb(total_traj, config)
