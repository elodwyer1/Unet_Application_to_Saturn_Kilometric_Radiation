from numba import jit, cuda
import pandas as pd
import numpy as np
from datetime import datetime
from ..read_config import config
import sqlite3
from tqdm import tqdm

def load_ephem_df():
    # Connect to the SQLite database
    conn = sqlite3.connect(f"{config.proc_data_fp}/segment_radio.db")
    
    # Load the data from the table into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM traj_df_allyears", conn)
    df['datetime_ut'] = pd.to_datetime(df['datetime_ut'])
    # Close the connection
    conn.close()
    
    return df

def load_ephem(dtime: datetime, dtime2: datetime):
    orbit_df = load_ephem_df()
    orbit_df = orbit_df.loc[orbit_df['datetime_ut'].between(dtime, dtime2), :]
    return orbit_df


@jit(nopython=True)
def normalize_data(lats: np.array, lts: np.array):
    # Normalize latitude to between 0 and 1
    nrlz_lats = (lats + 80) / 160
    
    # Normalize local time to between 0 and 1
    nrlz_lts = lts / 24
    
    return nrlz_lats, nrlz_lts

def interpolate_traj(start: datetime, end: datetime):
    # Load the trajectory data for the given time range
    traj_df_chunk = load_ephem(start, end)
    
    if traj_df_chunk.empty:
        return np.array([]), np.array([])
    
    # Convert Pandas Series to NumPy arrays
    lats = traj_df_chunk['LATITUDE'].to_numpy()
    lts = traj_df_chunk['LOCAL HOUR'].to_numpy()
    
    # Normalize data
    nrlz_lats, nrlz_lts = normalize_data(lats, lts)

    return nrlz_lats, nrlz_lts

@jit(nopython=True)
def median_absolute_deviation(data: np.array):
    # Calculate the median absolute deviation of the sample
    # Using nan-aware operations to handle missing values gracefully
    median = np.nanmedian(data)
    deviations = np.abs(data - median)
    mad = np.nanmedian(deviations)
    return mad, median

def take_median_std(start: datetime, end: datetime):
    # Interpolate the trajectory data
    nrlz_lats, nrlz_lts = interpolate_traj(start, end)

    if nrlz_lats.size == 0 or nrlz_lts.size == 0:
        return 0, 0, 0, 0
    
    # Calculate MAD and median for normalized local times
    lt_std, lt_med = median_absolute_deviation(nrlz_lts)
    # Calculate MAD and median for normalized latitudes
    lat_std, lat_med = median_absolute_deviation(nrlz_lats)
    
    
    return lt_med, lt_std, lat_med, lat_std


def main():
    try:
        if cuda.is_available():
            print("GPU detected. Using Numba for GPU-accelerated computations.")
        else:
            print("No GPU detected. Falling back to NumPy for computations.")
    except ImportError:
        print("Numba not installed. Falling back to NumPy for computations.")
    
    # Load and preprocess data
    df = pd.read_csv(config.output_data_fp + "/total_timestamps.csv", parse_dates=['start','end'])
    df_nosm = df.loc[df['label']!='LFE_sm', :].reset_index(drop=True)
    df_aug1 = pd.read_csv(config.output_data_fp + '/ML_lfeaug1_timestamps.csv', parse_dates=['start','end'], index_col=False)
    df_aug1 = df_aug1.loc[:,['start','end','label']]
    df_aug2 = pd.read_csv(config.output_data_fp + '/ML_lfeaug2_timestamps.csv', parse_dates=['start','end'], index_col=False)
    df_aug2 = df_aug2.loc[:,['start','end','label']]
    df_aug3 = pd.read_csv(config.output_data_fp + '/ML_lfeaug3_timestamps.csv', parse_dates=['start','end'], index_col=False).loc[:,['start','end']]
    df_aug3['label'] = 'LFE_aug3'
    df_aug4 = pd.read_csv(config.output_data_fp + '/ML_lfeaug4_timestamps.csv', parse_dates=['start','end'], index_col=False).loc[:,['start','end']]
    df_aug4['label'] = 'LFE_aug4'
    df_aug5 = pd.read_csv(config.output_data_fp + '/ML_lfeaug5_timestamps.csv', parse_dates=['start','end'], index_col=False).loc[:,['start','end']]
    df_aug5['label'] = 'LFE_aug5'
    total_df = pd.concat([df_nosm, df_aug1, df_aug2, df_aug3, df_aug4, df_aug5], axis=0)
    total_df.to_csv(config.output_data_fp + '/ML_total_timestamps_withaug.csv', index=False)
    
    vals = []
    # Process data in batches to avoid OOM
    batch_size = 10  # Adjust based on your memory capacity
    for i in tqdm(range(0, len(total_df), batch_size), desc="Processing batches"):
        batch = total_df.iloc[i:i + batch_size]
        batch_vals = []
        for start, end in zip(batch['start'], batch['end']):
            start = start.to_pydatetime()
            end = end.to_pydatetime()
            
            batch_vals.append(take_median_std(start, end))
        vals.extend(batch_vals)

    vals = np.asarray(vals)

    output_file_path = config.output_data_fp + '/ML_total_catalogue_withaug.csv'  # Update with your actual path
    total_df['lt_median'] = vals[:, 0]
    total_df['lt_stdev'] = vals[:, 1]
    total_df['lat_median'] = vals[:, 2]
    total_df['lat_stdev'] = vals[:, 3]
    total_df.to_csv(output_file_path, index=False)
    

if __name__ == "__main__":
    main()
