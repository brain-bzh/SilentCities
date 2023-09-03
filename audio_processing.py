import os
import torch
import numpy as np
import multiprocessing
import pandas as pd
import argparse
from tqdm import tqdm
from audioset_tagging_cnn.inference import audio_tagging,prepare_model
import utils.Dataset_site as dataset
import utils.utils
from analysis.tagging_validation import tagging_validate
#### a comment 
NUM_CORE = multiprocessing.cpu_count()
parser = argparse.ArgumentParser(
    description='Silent City Audio Tagging with pretrained LeeNet22 on Audioset')
parser.add_argument('--length', default=10, type=int, help='Segment length')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--metadata_folder', default=None,
                    type=str, help='folder with all metadata')
parser.add_argument('--site', default=None, type=str, help='site to process')
parser.add_argument('--folder', default=None, type=str,
                    help='Path to folder with wavefiles, will walk through subfolders')
parser.add_argument('--tomp3', default=None, type=str,
                    help='Path to folder for mp3 conversion, (Default : no conversion). Will create a subfolder with site name')
parser.add_argument('--database', default=None, type=str,
                    help='Path to metadata (given by metadata_site.py)')

parser.add_argument('--nocuda', action='store_false',
                    help='Do not use the GPU for acceleration')
parser.add_argument('--preload', action='store_true',
                    help='preload wave files into RAM')
parser.add_argument('--ncpu',default=NUM_CORE-4,type=int,
                    help='Number of CPUs for parallelization')

args = parser.parse_args()

if args.folder is None:
    raise(AttributeError("Must provide either a file or a folder"))
if args.metadata_folder is None:
    raise(AttributeError("Must provide metadata folder"))
if args.site is None:
    raise(AttributeError("Must provide site"))

mp3folder = args.tomp3
if not(mp3folder is None):
    print("Will convert to mp3...")
    mp3folder = os.path.join(args.tomp3,args.site)
    print(f"Creating folder {mp3folder}")
    os.makedirs(mp3folder,exist_ok=True)

checkpoint_path = 'ResNet22_mAP=0.430.pth'

if not(os.path.isfile(checkpoint_path)):
    raise(FileNotFoundError("Pretrained model {} wasn't found, did you download it ?".format(checkpoint_path)))

if args.database is None:
    raise(AttributeError("Must provide database"))

try:
    DATABASE = pd.read_pickle(args.database).reset_index(drop=True)
    if not(DATABASE.partID.dtype == 'int64'):
        raise('error part ID (must be int64) not {}'.format(DATABASE.partID.dtype))
except:
    DATABASE = pd.read_csv(args.database).reset_index(drop=True)
meta_site = pd.read_pickle(os.path.join(
    args.metadata_folder, args.site+'.pkl')).reset_index(drop=True)

print('preprocessing dataset')


audio_process_name = os.path.join(args.metadata_folder, '{}_process.pkl'.format(args.site))
csvfile = os.path.join(args.metadata_folder, 'results_{}.csv'.format(args.site))

if os.path.exists(audio_process_name):
    print(f"File exists, resuming processing, loading : {audio_process_name}")
    df_site = utils.utils.load_obj(audio_process_name)

else:
    df_site = {'name':[],'start':[], 'datetime': [], 'dB': [], 'ndsi_N': [], 'aci_N': [],
                    'BI_N': [], 'EAS_N': [],
                    'ECV_N': [], 'EPS_N': [] ,'ndsi_W': [], 'aci_W': [],
                    'BI_W': [], 'EAS_W': [],
                    'ECV_W': [], 'EPS_W': [], 'ACT':[],
                    'POWERB_126':[], 'POWERB_251':[], 'POWERB_501':[], 
                    'POWERB_1k':[], 'POWERB_2k':[], 'POWERB_4k':[], 
                    'POWERB_8k':[], 'POWERB_16k':[],
                    'clipwise_output':[], 'sorted_indexes' : [] ,'embedding' : []}
site_set,site_set_tagging = dataset.get_dataloader_site(
    args.site, args.folder, meta_site, df_site,args.metadata_folder,database = DATABASE, sr_eco=[48000,44100],sr_tagging=32000,batch_size=args.batch_size,mp3folder=mp3folder,ncpu=args.ncpu,preload=args.preload,rerun=True)
print('audio processing')

dotagging = True
if dotagging:
    print(f"Using CUDA : {args.nocuda}")
    device = torch.device('cuda') if (args.nocuda) and torch.cuda.is_available() else torch.device('cpu')

    print("Prepare model...")
    model = prepare_model(checkpoint_path , usecuda=args.nocuda,model_type="ResNet22")

    print("Audio Tagging...")
    for batch_idx, (inputs, info) in enumerate(tqdm(site_set_tagging)):

        with torch.no_grad():
            inputs = inputs.to(device)
            clipwise_output, labels, sorted_indexes, embedding = audio_tagging(inputs, model)
        
        for idx, date_ in enumerate(info['date']):
            df_site['clipwise_output'].append(clipwise_output[idx])
            df_site['sorted_indexes'].append(sorted_indexes[idx])
            df_site['embedding'].append(embedding[idx])
            df_site['datetime'].append(str(date_)) 
            df_site['name'].append(str(info['name'][idx]))
            df_site['start'].append(float(info['start'][idx]))
        
        if batch_idx%100 == 0:
            utils.utils.save_obj(df_site, audio_process_name)

    ## Saving the pkl with all the tagging
    utils.utils.save_obj(df_site, audio_process_name)
else:
    print("Skipping tagging - has results been loaded ? ")
## Generating the Dataframe with the subset of tagging categories as well as Geophony, Anthropophony and Biophony

Df_tagging = tagging_validate(df_site)

## Dataframe with only ecoacoustic indices and important metadata 
Df_eco = {'name':[],'start':[], 'datetime': [], 'dB': [], 'ndsi_N': [], 'aci_N': [],
                    'BI_N': [], 'EAS_N': [],
                    'ECV_N': [], 'EPS_N': [] ,'ndsi_W': [], 'aci_W': [],
                    'BI_W': [], 'EAS_W': [],
                    'ECV_W': [], 'EPS_W': [], 'ACT':[],
                    'POWERB_126':[], 'POWERB_251':[], 'POWERB_501':[], 
                    'POWERB_1k':[], 'POWERB_2k':[], 'POWERB_4k':[], 
                    'POWERB_8k':[], 'POWERB_16k':[]}


print("Eco acoustic indices and MP3 conversion...")
for batch_idx, info in enumerate(tqdm(site_set)):

    for idx, date_ in enumerate(info['date']):        
        Df_eco['datetime'].append(str(date_)) 
        Df_eco['name'].append(str(info['name'][idx]))
        Df_eco['start'].append(float(info['start'][idx]))
        for key in info['ecoac'].keys():
            Df_eco[key].append(float(info['ecoac'][key].numpy()[idx])) 

## Fusing with the dataframe containing only the ecoacoustic indices 

Df_final = pd.merge(Df_tagging,pd.DataFrame.from_dict(Df_eco),on=['name','start','datetime'])
Df_final.sort_values(by=['datetime','start']).to_csv(csvfile,index=False)