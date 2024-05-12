import pandas as pd
import numpy as np
import os
import wave

def get_bitdepth(filename):
    # Open the WAV file
    with wave.open(filename, 'rb') as wav_file:
        # Extract audio parameters
        sample_width = wav_file.getsampwidth()
    return sample_width * 8

import argparse

## argument parsing : we just need the path to the folder with all the db csv files
parser = argparse.ArgumentParser(description='Read all the dB files and add the dBfs column to the dataframes.')
parser.add_argument('path', type=str, help='path to the folder with all the db csv files')
parser.add_argument('output', type=str, help='path to the folder where the dBfs csv files will be saved')
## optional argument : path to the metadata csv file 
parser.add_argument('--metadata', type=str, default='/home/nfarrugi/Documents/datasets/silentcities/public/final_OSF/metadata/Site.csv',help='path to the metadata csv file')
## optional argument : path to the dataset folder
parser.add_argument('--datasetfolder', type=str, default='/bigdisk1/silentcities/',help='path to the dataset folder')
parser.add_argument('--datasetfolder2', type=str, default='/bigdisk2/silentcities/',help='path to the dataset folder')

args = parser.parse_args()

path = args.path

# create output folder if it does not exist
if not os.path.exists(args.output):
    os.makedirs(args.output,exist_ok=True)

## get all the csv files in the folder
files = [f for f in os.listdir(path) if f.endswith('.csv')]

## open the metadata file
metadata = pd.read_csv(args.metadata)

# create a new column for the bit depth and initialize it to 16 
metadata['bitdepth'] = 16

## open the files with pandas
constant = 20*np.log10(2**15/np.sqrt(2))
for f in files:
    data = pd.read_csv(os.path.join(path, f))
    data['dBfs'] = data['dB'] - constant

    ## check whether the obtained values are between -99 and 3
    if  (data['dBfs'] > 3).any():
        print(f'{f} has values out of the range [-99, 3]')
        print(data['dBfs'].min(), data['dBfs'].max())
        ## print the corresponding metadata (filename without extension is the part ID)
        print(metadata.loc[metadata['partID'] == f[:-4],'recorder_type'])

        # read a wave file from this site to get the bit depth 

        onefile = data.iloc[0]['filename']
        # search for this file in the site folders by doing a walk
        for root, dirs, files in os.walk(args.datasetfolder):
            if onefile in files:
                onefile = os.path.join(root, onefile)
                break
        for root, dirs, files in os.walk(args.datasetfolder2):
            if onefile in files:
                onefile = os.path.join(root, onefile)
                break
        try:
            bitdepth = get_bitdepth(onefile)
            #edit the metadata
            metadata.loc[metadata['partID'] == f[:-4],'bitdepth'] = bitdepth
            print(f'bitdepth is {bitdepth}, recalculating dBfs...')
            constant = 20*np.log10(2**(bitdepth - 1)/np.sqrt(2))
            data['dBfs'] = data['dB'] - constant
        except:
            print(f'Error when opening file {onefile} to get the bitdepth')

            continue

    ## write the new file in the output folder, with the same name
    data.to_csv(os.path.join(args.output, f), index=False)

metadata.to_csv(args.metadata, index=False)