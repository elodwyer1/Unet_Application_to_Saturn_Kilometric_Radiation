import pandas as pd
import numpy as np
from datetime import datetime
import math
from tfcat import TFCat
from astropy.time import Time
from ..read_config import config

#get co-ordinates of each item in json file of polygons
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
    #dataframe of sorted values of start, end and label of each LFE
    df = pd.DataFrame({'start': start, 'end':end,'label':feature})
    df = df.drop_duplicates(subset='start', keep='first')
    df=df.sort_values(by='start').reset_index(drop=True)

    return df

#function that takes a list of start and end times of LFEs and finds the gaps between labelled LFEs.
#It then splits the gaps into chunks, with length defined by 'val' parameter.
def extend_times(start, end, val):
    s = list(end[:-1])
    e=list(start[1:])
    #duration of time between end and start of subsequent LFEs
    dur = np.array([(j-i).total_seconds()/3600 for i, j in zip(s,e)])
    #dataframe wth start, end and duration of each gap.
    df = pd.DataFrame({'start': s, 'end':e,'dur': dur})
    #find gaps longer than val.
    df=df.loc[df['dur']>val, :].reset_index(drop=True)
    #find x, dur=val * x
    multiple=df['dur'].apply(lambda x: math.floor(x/val))
    start=df['start']
    startlist=[]
    #loop through each gap between LFEs and find start times of each 5 hr chunk.
    for i in range(len(start)):
        s=start[i]
        j=0
        while j < multiple[i]: 
            startlist.append(s)
            s=s+pd.Timedelta(val,'h')
            j+=1
    endlist = [i+pd.Timedelta(val, 'h') for i in startlist]
    #start and end times of each section of non-LFE 
    return startlist, endlist

def make_catalogue_csv_files():
    '''Make dataframe of start time of LFE, end time of LFE and label.'''
    #Open the file and extract co-ordinates for each feature, as well as feature label.
    file= config.input_data_fp + "/SKR_LFEs.json"
    lfe_df=make_dataframe(file)
    lfe_df.to_csv(config.output_data_fp + "/lfe_timestamps.csv",index=False)
    ''' Use fully labelled sections of data to find intervals labelled as 'non-LFE' '''

    #2006 is fully labelled
    #Find start and end times of LFEs in 2006
    lfe_df_061 = lfe_df.loc[lfe_df['start'].between(pd.Timestamp('20060101'), pd.Timestamp('20070101')),:].reset_index(drop=True)
    #Split data without LFE into 5 hr long sections and return start and end times.
    start_nolfe061, end_nolfe061 = extend_times(lfe_df_061['start'],lfe_df_061['end'], 5)

    #March, May and December of 2008 are fully labelled.
    #Find start and end times of LFEs in March, May and December 2008
    lfe_df_dec08 = lfe_df.loc[lfe_df['start'].between(pd.Timestamp('20081201'), pd.Timestamp('20090101')),:].reset_index(drop=True)
    lfe_df_mar08= lfe_df.loc[lfe_df['start'].between(pd.Timestamp('20080301'), pd.Timestamp('20080401')),:].reset_index(drop=True)
    lfe_df_may08= lfe_df.loc[lfe_df['start'].between(pd.Timestamp('20080501'), pd.Timestamp('20080601')),:].reset_index(drop=True)

    #Split data without LFE into 5 hr long sections and return start and end times.
    start_nolfe081, end_nolfe081 = extend_times(lfe_df_mar08['start'], lfe_df_mar08['end'], 5)
    start_nolfe082, end_nolfe082 = extend_times(lfe_df_may08['start'], lfe_df_may08['end'], 5)
    start_nolfe083, end_nolfe083 = extend_times(lfe_df_dec08['start'], lfe_df_dec08['end'], 5)

    #Combine 2006 and 2008 values.
    start_nolfe = np.concatenate([start_nolfe061, start_nolfe081, start_nolfe082, start_nolfe083], axis=0)
    end_nolfe = np.concatenate([end_nolfe061,end_nolfe081, end_nolfe082, end_nolfe083],axis=0)
    label = np.repeat('NoLFE', len(start_nolfe))
    #dataframe with columns: 'start', 'end' and 'label'.
    nolfe_df = pd.DataFrame({'start': start_nolfe, 'end': end_nolfe, 'label':label})
    nolfe_df.to_csv(config.output_data_fp + "/nolfe_timestamps.csv",index=False)
    #dataframe with columns: 'start', 'end' and 'label' for all LFEs and non-LFEs
    total_df=pd.concat([lfe_df,nolfe_df],axis=0).reset_index(drop=True)
    total_df.to_csv(config.output_data_fp + "/total_timestamps.csv",index=False)