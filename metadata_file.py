import os
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
from tqdm import tqdm

import librosa
import utils.utils as utils
import argparse


parser = argparse.ArgumentParser(description='Silent City Meta data genetor')

parser.add_argument('--folder', default=None, type=str, help='Path to folder with wavefiles, will walk through subfolders')
parser.add_argument('--save_path', default=None, type=str, help='Path to save meta data')
parser.add_argument('--verbose', action='store_true', help='Verbose (default False = nothing printed)')
#parser.add_argument('--all', action='store_true', help='process all site. for dev')

args = parser.parse_args()
HDD =  args.folder
# if args.all:
#     HDD = ['/media/nicolas/Silent', '/media/nicolas/LaCie'] # list hdd if process all file and multiple HDD

# if args.folder is None and not(args.all):    
#     raise(AttributeError("Must provide either a file or a folder"))
    
if not(os.path.exists(args.save_path)):
    os.mkdir(args.save_path)

def metadata_generator(folder):
    '''
    Generate meta data for one folder (one site) and save in csv and pkl

    '''
    if folder[-1] == '/'  : folder = folder.pop(-1)
    save_name = os.path.join(args.save_path, os.path.basename(folder))

    filelist = []

    if os.path.exists( save_name+'.pkl'):
        Df = pd.read_pickle( save_name+'.pkl')
        Df_error = pd.read_pickle( save_name+'_error.pkl')
    else:
        Df = pd.DataFrame(columns=['filename', 'datetime', 'length', 'sr', 'dB'])
        Df_error = pd.DataFrame(columns=['filename','error'])

    filename_ = [Df['filename'][k] for k in range(len(Df))]
    [filename_.append(Df_error['filename'][k]) for k in range(len(Df_error))]
    if HDD == folder:
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                if name[-3:].casefold() == 'wav' and name[:2] != '._':
                    if args.verbose:
                        print(os.path.join(root, name))
                    filelist.append(os.path.join(root, name))
    else:
        for hdd_path in HDD:
            for root, dirs, files in os.walk(os.path.join(hdd_path, folder), topdown=False):
                for name in files:
                    if name[-3:].casefold() == 'wav' and name[:2] != '._':
                        filelist.append(os.path.join(root, name))
        
    for idx, wavfile in tqdm(enumerate(filelist)):
        
        if (os.path.basename(wavfile) in filename_) == False:
            try :     
                _, meta = utils.read_audio_hdr(wavfile, verbose=args.verbose) #meta data
                try:
                    sr, x = wav.read(wavfile)
                except:
                    x, sr = librosa.load(wavfile, sr = None)
                    x = x*2**15

                if len(x)>1:
                    Dfwav = pd.DataFrame.from_dict({'datetime': meta['datetime'], 'filename': os.path.basename(wavfile), 'length' : [len(x)], 'sr' : [sr], 'dB' : [20*np.log10(np.std(x))]})
                    Df = pd.concat([Df,Dfwav], ignore_index=True)
                else:
                    Df_error = pd.concat([Df_error,pd.DataFrame.from_dict({'filename': os.path.basename(wavfile), "error": "Empty wave file"})], ignore_index=True)
            except Exception as e:
                try:
                    print(e) 
                    Df_error = pd.concat([Df_error,pd.DataFrame({'filename': os.path.basename(wavfile), "error": e})], ignore_index=True)
                except :
                    pass
            
            if idx%1000 == 0:
                Df.to_pickle(save_name+'.pkl')
                Df_error.to_pickle(save_name+'_error.pkl')
        
    if args.verbose:
        print(Df)

    Df = Df.sort_values('datetime')
    Df.to_csv(save_name+'.csv')
    Df.to_pickle(save_name+'.pkl')
    Df_error.to_pickle(save_name+'_error.pkl')

# if args.all :
#     for k in range(1,400):
#         print('{:04d}'.format(k))
#         metadata_generator('{:04d}'.format(k))
# else :

metadata_generator(args.folder)



