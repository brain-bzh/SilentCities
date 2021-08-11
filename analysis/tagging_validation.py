#!/usr/bin/env python
# coding: utf-8

import os 
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd 
import argparse
import datetime
import pickle 
import csv 

### List of labels to keep

fewlabels = ['Bird vocalization, bird call, bird song',
'Caw','Crow','Owl','Grunt','Speech','Bee, wasp, etc.','Cricket','Rain','Thunderstorm','Aircraft','Helicopter','Car']

def linear(Df):

    Df['press'] = np.exp(Df.dB*(np.log(10)/20))
    return Df

# Load label
with open('../audioset_tagging_cnn/class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)


def return_preprocessed(Df):
    # data
    allprobas = np.stack(Df['clipwise_output'])
    labelmax = np.argmax(allprobas,axis=1)
    allembed = np.stack(Df['embedding'])        

    return allprobas,labelmax,allembed

def subset_probas(Df,search_labels):
    allprobas,_,_ = return_preprocessed(Df)

    ind_list = []
    for curlabel in search_labels:
        ind_list.append(int(np.argwhere([c==curlabel for c in labels])))

    return allprobas[:,ind_list]    

# Audio Tagging

def load_obj_tag(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)


parser = argparse.ArgumentParser(description='Script to test ecoacoustic indices parameters')

parser.add_argument('--input', default=None, type=str, help='Path to pkl file')
parser.add_argument('--save_path', default='/bigdisk2/meta_silentcities/tests_eco', type=str, help='Path to save output csv and pkl')

args = parser.parse_args()

site= str.split(args.input,sep='_')[-1].split(sep='.')[0]
savepath = args.save_path
CSV_SAVE = os.path.join(savepath,f'average_tagging_site_{site}.csv')
pkl_save = os.path.join(savepath,f'average_tagging_site_{site}.pkl')
figfile  = os.path.join(savepath,f'average_tagging_figure_{site}.html')

Df = load_obj_tag(args.input)

probas = subset_probas(Df,fewlabels)

# create Dataframe from Dict

Df_new = pd.DataFrame()
for i,curlab in enumerate(fewlabels):
    Df_new[curlab] = probas[:,i]
### Saving 

Df_new.to_csv(CSV_SAVE)