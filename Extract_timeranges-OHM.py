#!/usr/bin/env python
# coding: utf-8

import os 
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd 


def linear(Df):

    Df['press'] = np.exp(Df.dB*(np.log(10)/20))
    return Df


# Utility functions 
# --


import datetime
def extract_timerange(Df,start=None,end=None):
    """
    Extracts a specific time range for a Dataframe with a datetime column 
    
    arguments :
    Df : DataFrame, 
    start : defined with Timestamp, for ex pd.Timestamp('2020-03-16 06:30:00')
    end : defined with Timestamp, for ex pd.Timestamp('2020-03-16 06:30:00')
    
    outputs : 
    
    Df_extract : Extraction at this range

    """
    
    #Let's use the datetime (Timestamp) as a new index for the dataframe
    
    ts = pd.DatetimeIndex(Df.datetime)
    Df.index = ts

    
    if start < min(ts):
        start=ts[0]
    if end > max(ts):
        end=ts[-1]

    return Df.loc[start:end]



import pickle 

def load_obj(name):
    with open(name , 'rb') as f:
        Df = pd.DataFrame.from_dict(pickle.load(f))
        for idx, k in enumerate(Df['datetime']) :
            Df['datetime'][idx] = datetime.datetime.strptime(k, '%Y%m%d_%H%M%S')
        return linear(Df)



def extract_daterange_attime(Df,startday=None,endday=None,startime="00:00:00",endtime="05:00:00"):
    
    if startday is None:
        startday = min(Df['datetime'])
        
    if endday is None:
        endday = max(Df['datetime'])
    
    ## define the days 
    days = pd.period_range(start=startday,end=endday,freq='D')
    
    allDf = []
    
    for curday in days:
        day,month,year = curday.day,curday.month,curday.year
        
        st = pd.Timestamp(f"{year}-{month}-{day} {startime}")
        ed = pd.Timestamp(f"{year}-{month}-{day} {endtime}")
        
        allDf.append(extract_timerange(Df,start=st,end=ed))
        
    if len(allDf)== 0:
        return pd.DataFrame([])
    else:
        return pd.concat(allDf)                            



def extract_daterange_eachhour(Df,hour_start,startday=None,endday=None):
    
    
    Df_extr = (extract_daterange_attime(Df,startday=startday,endday=endday,startime=f"{hour_start}:00",endtime=f"{hour_start}:09")).mean().to_frame(name=f"{hour_start}:00")
    Df_extr2 = (extract_daterange_attime(Df,startday=startday,endday=endday,startime=f"{hour_start}:10",endtime=f"{hour_start}:19")).mean().to_frame(name=f"{hour_start}:10")
    Df_extr3 = (extract_daterange_attime(Df,startday=startday,endday=endday,startime=f"{hour_start}:20",endtime=f"{hour_start}:29")).mean().to_frame(name=f"{hour_start}:20")
    Df_extr4 = (extract_daterange_attime(Df,startday=startday,endday=endday,startime=f"{hour_start}:30",endtime=f"{hour_start}:39")).mean().to_frame(name=f"{hour_start}:30")
    Df_extr5 = (extract_daterange_attime(Df,startday=startday,endday=endday,startime=f"{hour_start}:40",endtime=f"{hour_start}:49")).mean().to_frame(name=f"{hour_start}:40")
    Df_extr6 = (extract_daterange_attime(Df,startday=startday,endday=endday,startime=f"{hour_start}:50",endtime=f"{hour_start}:59")).mean().to_frame(name=f"{hour_start}:50")

    return pd.concat([Df_extr,Df_extr2,Df_extr3,Df_extr4,Df_extr5,Df_extr6],axis=1).transpose().sort_index().drop(columns=['index','start'])


# In[11]:


def extract_av_acrossdays(Df):
    Dflist = []
    for hours in range(24):
        for quart in [0,15,30,45]:
            for minutes in range(5):
                hour_start = "{:02d}:{:02d}".format(hours,minutes+quart)
                print(hour_start)
                Dflist.append(extract_daterange_eachhour(Df,hour_start))


    
    return pd.concat(Dflist)



import argparse

parser = argparse.ArgumentParser(description='Script to test ecoacoustic indices parameters')

parser.add_argument('--site', default=None, type=str, help='Which site to process')
parser.add_argument('--input', default=None, type=str, help='Path to pkl file')
parser.add_argument('--save_path', default='/bigdisk2/meta_silentcities/tests_eco', type=str, help='Path to save output csv and pkl')

args = parser.parse_args()

site= args.site
savepath = args.save_path
CSV_SAVE = os.path.join(savepath,f'average_site_{site}.csv')
pkl_save = os.path.join(savepath,f'average_site_{site}.pkl')
figfile  = os.path.join(savepath,f'average_figure_{site}.html')

Df = load_obj(args.input)
df_site = extract_av_acrossdays(Df)

df_site['db'] = 20*np.log10(df_site['press'])

df_site.to_csv(CSV_SAVE)
df_site.to_pickle(pkl_save)


import plotly.graph_objects as go
from plotly.subplots import make_subplots


df_site['datetime'] = df_site.index

for idx, k in enumerate(df_site['datetime']) :
    df_site['datetime'][idx] = datetime.datetime.strptime(k, '%H:%M:%S')
        

indic = ['db', 'ndsi', 'aci', 'nbpeaks', 'BI', 'EAS', 'ECV', 'EPS']

fig = make_subplots(rows=5, cols=2,print_grid=True, subplot_titles=indic,shared_xaxes='all')
for idx, k in enumerate(indic): 
    fig.add_trace(go.Scatter(x=df_site['datetime'], y=df_site[k]),row=(idx//2)+1, col=(idx%2)+1)

fig.update(layout_showlegend=False)
fig.write_html(figfile)