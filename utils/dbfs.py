import pandas as pd
import numpy as np
import os

import argparse

## argument parsing : we just need the path to the folder with all the db csv files
parser = argparse.ArgumentParser(description='Read all the dB files and add the dBfs column to the dataframes.')
parser.add_argument('path', type=str, help='path to the folder with all the db csv files')
parser.add_argument('output', type=str, help='path to the folder where the dBfs csv files will be saved')
## optional argument : path to the metadata csv file 
parser.add_argument('--metadata', type=str, default='/home/nfarrugi/Documents/datasets/silentcities/public/final_OSF/metadata/Site.csv',help='path to the metadata csv file')
args = parser.parse_args()

path = args.path

# create output folder if it does not exist
if not os.path.exists(args.output):
    os.makedirs(args.output,exist_ok=True)

## get all the csv files in the folder
files = [f for f in os.listdir(path) if f.endswith('.csv')]

## open the metadata file
metadata = pd.read_csv(args.metadata)

## open the files with pandas
constant = 20*np.log10(2**15/np.sqrt(2))
for f in files:
    data = pd.read_csv(os.path.join(path, f))
    data['dBfs'] = data['dB'] - constant

    ## check whether the obtained values are between -99 and 3
    if (data['dBfs'] < -99).any() or (data['dBfs'] > 3).any():
        print(f'{f} has values out of the range [-99, 3]')
        print(data['dBfs'].min(), data['dBfs'].max())
        ## print the corresponding metadata (filename without extension is the part ID)
        print(metadata.loc[metadata['partID'] == f[:-4],'recorder_type'])
        continue

    ## write the new file in the output folder, with the same name
    data.to_csv(os.path.join(args.output, f), index=False)
    