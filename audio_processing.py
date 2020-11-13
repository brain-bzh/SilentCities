import os
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from audioset_tagging_cnn.inference import audio_tagging
import utils.Dataset_site as dataset
import utils.utils

parser = argparse.ArgumentParser(
    description='Silent City Audio Tagging with pretrained LeeNet22 on Audioset')
parser.add_argument('--length', default=10, type=int, help='Segment length')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--metadata_folder', default=None,
                    type=str, help='folder with all metadata')
parser.add_argument('--site', default=None, type=str, help='site to process')
parser.add_argument('--folder', default=None, type=str,
                    help='Path to folder with wavefiles, will walk through subfolders')
parser.add_argument('--database', default=None, type=str,
                    help='Path to metadata (given by metadata_site.py)')

parser.add_argument('--nocuda', action='store_false',
                    help='Do not use the GPU for acceleration')

args = parser.parse_args()

if args.folder is None:
    raise(AttributeError("Must provide either a file or a folder"))
if args.metadata_folder is None:
    raise(AttributeError("Must provide metadata folder"))
if args.site is None:
    raise(AttributeError("Must provide site"))

checkpoint_path = 'ResNet22_mAP=0.430.pth'

if not(os.path.isfile(checkpoint_path)):
    raise(FileNotFoundError("Pretrained model {} wasn't found, did you download it ?".format(checkpoint_path)))

if args.database is None:
    raise(AttributeError("Must provide database"))

DATABASE = pd.read_pickle(args.database).reset_index(drop=True)
if not(DATABASE.partID.dtype == 'int64'):
    raise('error part ID (must be int64) not {}'.format(DATABASE.partID.dtype))

meta_site = pd.read_pickle(os.path.join(
    args.metadata_folder, args.site+'.pkl')).reset_index(drop=True)

print('preprocessing dataset')


audio_process_name = os.path.join(args.metadata_folder, '{}_process.pkl'.format(args.site))

if os.path.exists(audio_process_name):
    df_site = utils.utils.load_obj(audio_process_name)

else:
    df_site = {'name':[],'start':[], 'datetime': [], 'ndsi': [], 'aci': [], 'nbpeaks': [] , 'BI' : [], 'EVN' : [], 'ACT' : [], 'EAS':[], 'ECV' : [], 'EPS' : [],
    'ndsi_filt': [], 'aci_filt': [], 'nbpeaks_filt': [] , 'BI_filt' : [], 'EVN_filt' : [], 'ACT_filt' : [], 'EAS_filt':[], 'ECV_filt' : [], 'EPS_filt' : [],
                                'clipwise_output':[], 'sorted_indexes' : [] ,'embedding' : []}
site_set = dataset.get_dataloader_site(
    args.site, args.folder, meta_site, df_site,args.metadata_folder,database = DATABASE, batch_size=args.batch_size)
print('audio processing')

for batch_idx, (inputs, info) in tqdm(enumerate(site_set)):
    
    
    

    with torch.no_grad():

        clipwise_output, labels, sorted_indexes, embedding = audio_tagging(inputs, checkpoint_path , usecuda=args.nocuda)
    
    for idx, date_ in enumerate(info['date']):
        df_site['clipwise_output'].append(clipwise_output[idx])
        df_site['sorted_indexes'].append(sorted_indexes[idx])
        df_site['embedding'].append(embedding[idx])
        df_site['datetime'].append(str(date_)) 
        df_site['name'].append(str(info['name'][idx]))
        df_site['start'].append(float(info['start'][idx]))
        for key in info['ecoac'].keys():
            df_site[key].append(float(info['ecoac'][key].numpy()[idx])) 
    # print(df_site['start'])
    
    if batch_idx%100 == 0:
        utils.utils.save_obj(df_site, audio_process_name)
utils.utils.save_obj(df_site, audio_process_name)
    
# print(df_site)20424
