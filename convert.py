import os
import torch
import numpy as np
import multiprocessing
import pandas as pd
import argparse
from tqdm import tqdm
from audioset_tagging_cnn.inference import audio_tagging,prepare_model
import utils.utils
from analysis.tagging_validation import tagging_validate
from utils.parameters import len_audio_s
import datetime
import librosa
import tempfile
from torch.utils.data import Dataset
from pydub import AudioSegment
import soundfile as sf
#python convert.py --metadata_folder /bigdisk1/meta_silentcities/dbfiles --site 0065 --folder /bigdisk2/silentcities/0065/ --database /bigdisk1/database_pross.pkl --batch_size 16 --toflac /bigdisk1/flac
defult_tmp_dir = tempfile._get_default_tempdir()
#### a comment 
NUM_CORE = multiprocessing.cpu_count()
parser = argparse.ArgumentParser(
    description='Converting files to FLAC')
parser.add_argument('--length', default=10, type=int, help='Segment length')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--metadata_folder', default=None,
                    type=str, help='folder with all metadata')
parser.add_argument('--site', default=None, type=str, help='site to process')
parser.add_argument('--folder', default=None, type=str,
                    help='Path to folder with wavefiles, will walk through subfolders')
parser.add_argument('--toflac', default=None, type=str,
                    help='Path to folder for FLAC conversion, (Default : no conversion). Will create a subfolder with site name')
parser.add_argument('--database', default=None, type=str,
                    help='Path to metadata (given by metadata_site.csv)')

parser.add_argument('--nocuda', action='store_false',
                    help='Do not use the GPU for acceleration')
parser.add_argument('--preload', action='store_true',
                    help='preload wave files into RAM')
parser.add_argument('--ncpu',default=NUM_CORE-1,type=int,
                    help='Number of CPUs for parallelization')

args = parser.parse_args()


## utility functions


class Convert_Dataset(Dataset):
    def __init__(self, meta_dataloader,toflac,preload=False):
        self.meta = meta_dataloader
        self.toflac = toflac ### this must be the output site folder
        print(f"Temp folder for wav files for mp3 conversion : {defult_tmp_dir}")
        self.preload = preload
        self.data = []
        if preload:
            print('Preloading dataset...')
            for idx,curfile in tqdm(enumerate(self.meta['filename'])):
                filename = curfile

                wav_o, sr = librosa.load(filename, sr=None, mono=True,offset=self.meta['start'][idx], duration=len_audio_s)
                self.data.append((wav_o,sr))


    def __getitem__(self, idx):
        filename = self.meta['filename'][idx]

        if self.preload:
            wav_o,sr = self.data[idx]
        else:
            wav_o, sr = librosa.load(filename, sr=None, mono=True,
                              offset=self.meta['start'][idx], duration=len_audio_s)        
        
        ## name of the mp3 file
        flacfile = os.path.join(self.toflac,f"{os.path.splitext(os.path.split(filename)[1])[0]}_{self.meta['start'][idx]}.flac")
        #print(flacfile)
        if not(os.path.isfile(flacfile)):
            #print(f"Converting {flacfile}...")                
            ## Converting current chunk to temporary wav file in a temp folder

            temp_name = os.path.join(defult_tmp_dir,next(tempfile._get_candidate_names()) + '.wav')
            sf.write(temp_name,wav_o,sr)

            ## reading  chunk
            wav_audio = AudioSegment.from_file(temp_name, format="wav")

            file_handle = wav_audio.export(flacfile,
                    format="flac")

            os.remove(temp_name)
        
        
        return {'name': os.path.basename(filename), 'start': self.meta['start'][idx],
                                                        'date': self.meta['date'][idx].strftime('%Y%m%d_%H%M%S')}

    def __len__(self):
        return len(self.meta['filename'])
    



def get_dataloader_site(site_ID, path_wavfile, meta_site, meta_path, database, batch_size=6,flacfolder = None,ncpu=NUM_CORE,preload=False):
    partIDidx = database[database.partID == int(site_ID)].index[0]
    file_refdB = database['ref_file'][partIDidx]

    meta_dataloader = pd.DataFrame(
        columns=['filename', 'sr', 'start', 'stop'])

    filelist = []
    for root, dirs, files in os.walk(path_wavfile, topdown=False):
        for name in files:
            if name[-3:].casefold() == 'wav' and name[:2] != '._':
                filelist.append(os.path.join(root, name))
    
    filelist_base = [os.path.basename(file_) for file_ in filelist]

    for idx, wavfile in enumerate(tqdm(meta_site['filename'])):

        len_file = meta_site['length'][idx]
        sr_in = meta_site['sr'][idx]

        duration = len_file/sr_in

        nb_win = int(duration // len_audio_s )

        for win in range(nb_win):
            delta = datetime.timedelta(seconds=int((win*len_audio_s)))
            curmeta = pd.DataFrame.from_dict({'filename': [filelist[filelist_base.index(wavfile)]], 'sr': [sr_in], 'start': (
                [win*len_audio_s]), 'stop': [((win+1)*len_audio_s)], 'len': [len_file], 'date': meta_site['datetime'][idx] + delta})
            meta_dataloader = pd.concat([meta_dataloader,curmeta], ignore_index=True)
        
        if wavfile == file_refdB:
            file_refdB = filelist[filelist_base.index(wavfile)]
        
    meta_dataloader.to_pickle(os.path.join(meta_path, site_ID+'_metaloader.pkl'))
    print(meta_dataloader)

    site_set = Convert_Dataset(meta_dataloader=meta_dataloader.reset_index(drop=True),
    toflac=flacfolder,preload=preload)
    site_set = torch.utils.data.DataLoader(
        site_set, batch_size=batch_size, shuffle=False, num_workers=ncpu)

    return site_set

if args.folder is None:
    raise(AttributeError("Must provide either a file or a folder"))
if args.metadata_folder is None:
    raise(AttributeError("Must provide metadata folder"))
if args.site is None:
    raise(AttributeError("Must provide site"))

flacfolder = args.toflac
if not(flacfolder is None):
    print("Will convert to FLAC...")
    flacfolder = os.path.join(args.toflac,args.site)
    print(f"Creating folder {flacfolder}")
    os.makedirs(flacfolder,exist_ok=True)

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

#(site_ID, path_wavfile, meta_site, meta_path, database, batch_size=6,flacfolder = None,ncpu=NUM_CORE,preload=False)
site_set = get_dataloader_site(
    site_ID=args.site, path_wavfile=args.folder, meta_site=meta_site,meta_path=args.metadata_folder,database = DATABASE,batch_size=args.batch_size,flacfolder=flacfolder,ncpu=args.ncpu,preload=args.preload)

print("FLAC conversion...")


for batch_idx, info in enumerate(tqdm(site_set)):

    pass

print("Done !")