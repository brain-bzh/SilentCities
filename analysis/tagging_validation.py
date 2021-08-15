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



## opening class list 
Df_classes = pd.read_csv('analysis/TaggingCategory.csv')

dict_allcats = {}
fewlabels = []
for curcat in Df_classes.columns:
    list_curcat = []
    for idx, curclass in enumerate(Df_classes[curcat]):
        if curclass is not np.nan:
            list_curcat.append(curclass)
            fewlabels.append(curclass)
    
    dict_allcats[curcat] = list_curcat

def linear(Df):

    Df['press'] = np.exp(Df.dB*(np.log(10)/20))
    return Df

# Load label
with open('audioset_tagging_cnn/class_labels_indices.csv', 'r') as f:
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
    newlabellist = []
    notfound = []
    for curlabel in search_labels:
        try:
            ind_list.append(int(np.argwhere([c==curlabel for c in labels])))
            newlabellist.append(curlabel)
        except:
            print(f"Label {curlabel} not present in PANN training")
            notfound.append(curlabel)

    return allprobas[:,ind_list],newlabellist,notfound   

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
CSV_SAVE = os.path.join(savepath,f'tagging_site_{site}.csv')

Df = load_obj_tag(args.input)
probas,newlabellist,notfound = subset_probas(Df,fewlabels)

fewlabels = newlabellist

### Lets first generate a csv with all the probabilities, no grouping
# 
# # create Dataframe from Dict

Df_new = pd.DataFrame()
Df_new['name'] = Df['name']
Df_new['start'] = Df['start']
Df_new['datetime'] = Df['datetime']

#for i,curlab in enumerate(fewlabels):
#    Df_new[curlab] = probas[:,i]
### Saving 


### Now let s generate another csv to have the matching with categories of the manual identification protocol
### to do that, we take the maximum probability in the subcategory identified by the dictionnary 
### We parse using the dictionnary dict_allcats

for cursubcat in dict_allcats.keys():
    curlabels = dict_allcats[cursubcat]

    probas_sub,newlabellist_sub,_ = subset_probas(Df,curlabels)

    probas_max = np.max(probas_sub,axis=1)

    cursublabel = 'tag_' + cursubcat

    Df_new[cursublabel] = probas_max

### And Finally let's do an estimation of BioPhony, Antropophony and Geophony level using audio tagging results 
### For that, we group the categories accordingly : 

macro_cat = {'geophony':['Wind', 'Rain', 'River', 'Wave', 'Thunder'],'biophony': ['Bird', 'Amphibian', 'Insect', 'Mammal', 'Reptile'], 'anthropophony': ['Walking', 'Cycling', 'Beep', 'Car', 'Car honk', 'Motorbike', 'Plane', 'Helicoptere', 'Boat', 'Others_motors', 'Shoot', 'Bell', 'Talking', 'Music', 'Dog bark', 'Kitchen sounds', 'Rolling shutter']}


### and now we will calculate the average probability in each of the three macro categories 

for cursubcat in macro_cat.keys():
    curlabels = ['tag_' + i for i in macro_cat[cursubcat]]

    curDf = Df_new[curlabels]

    Df_new[cursubcat] = curDf.max(axis=1)

Df_new.sort_values(by=['datetime','start']).to_csv(CSV_SAVE,index=False)