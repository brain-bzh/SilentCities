import os
import multiprocessing
from tqdm import tqdm
import datetime

import librosa
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal as sig
from scipy.signal import resample

import torch
from torch.utils.data import Dataset

from utils import utils
from utils.ecoacoustics import compute_NDSI, compute_NB_peaks, compute_ACI, compute_spectrogram
from utils.alpha_indices import acoustic_events, acoustic_activity, spectral_entropy, bioacousticsIndex
from scipy.signal import butter, filtfilt
from utils.parameters import len_audio_s
from utils.utils import save_obj

from audioset_tagging_cnn.inference import audio_tagging
checkpoint_path = 'ResNet22_mAP=0.430.pth'

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)

    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


class Silent_dataset(Dataset):
    def __init__(self, meta_dataloader,  Fmin, Fmax, refdB):
        self.ref_dB = refdB                
        self.meta = meta_dataloader
        self.Fmin, self.Fmax = Fmin, Fmax

    def __getitem__(self, idx):

        filename = self.meta['filename'][idx]

        wav, sr = librosa.load(filename, sr=None, mono=True,
                              offset=self.meta['start'][idx], duration=len_audio_s)

        wav = resample(wav, int(len_audio_s*32000))
        wav = torch.FloatTensor(wav)

        return wav,{'name': os.path.basename(filename),'start': self.meta['start'][idx],
                                                        'date': self.meta['date'][idx].strftime('%Y%m%d_%H%M%S')}
    def __len__(self):
        return len(self.meta['filename'])
    
def get_dataloader_site(path_wavfile, meta_site,Fmin, Fmax, batch_size=1):

    meta_dataloader = pd.DataFrame(
        columns=['filename', 'sr', 'start', 'stop'])

    for idx, wavfile in enumerate(tqdm(meta_site['filename'])):

        len_file = meta_site['length'][idx]
        sr_in = meta_site['sr'][idx]
        duration = len_file/sr_in
        nb_win = int(duration // len_audio_s )

        for win in range(nb_win):
            delta = datetime.timedelta(seconds=int((win*len_audio_s)))
            meta_dataloader = meta_dataloader.append({'filename': wavfile, 'sr': sr_in, 'start': (
                win*len_audio_s), 'stop': ((win+1)*len_audio_s), 'len': len_file, 'date': meta_site['datetime'][idx] + delta}, ignore_index=True)
        # if duration % len_audio_s == float(0):
        #     delta = datetime.timedelta(seconds=int((nb_win-1)*len_audio_s))
        #     meta_dataloader = meta_dataloader.append({'filename': filelist[filelist_base.index(wavfile)], 'sr': sr_in, 'start': (
        #         duration - len_audio_s), 'stop': (duration), 'len': len_file, 'date': meta_site['datetime'][idx] + delta}, ignore_index=True)

    site_set = Silent_dataset(meta_dataloader.reset_index(drop=True),Fmin, Fmax, ref_dB)
    site_set = torch.utils.data.DataLoader(
        site_set, batch_size=batch_size, shuffle=False, num_workers=NUM_CORE)

    return site_set


def metadata_generator(folder):
    '''
    Generate meta data for one folder (one site) and save in csv and pkl

    '''

    filelist = []
    Df = pd.DataFrame(columns=['filename', 'datetime', 'length', 'sr'])
    Df_error = pd.DataFrame(columns=['filename'])

    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            if name[-3:].casefold() == 'wav' and name[:2] != '._':
                filelist.append(os.path.join(root, name))
        
    for idx, wavfile in enumerate(tqdm(filelist)):
        _, meta = utils.read_audio_hdr(wavfile, False) #meta data

        # if idx == 0 : 
        try:
            sr, x = wav.read(wavfile)
        except:
            print('skipping short file')
            continue

        Df = Df.append({'datetime': meta['datetime'], 'filename': wavfile, 'length' : len(x), 'sr' : sr}, ignore_index=True)
            
    Df = Df.sort_values('datetime')
    return(Df)



def compute_ecoacoustics(wavforme,sr, ref_mindb, Fmin, Fmax):

    
    #wavforme = butter_bandpass_filter(wavforme, Fmin, Fmax, fs = sr)

    Sxx, freqs = compute_spectrogram(wavforme, sr)

    Sxx_dB = 10*np.log10(Sxx)
    N = len(wavforme)
    dB = 20*np.log10(np.std(wavforme))
    
    nbpeaks = compute_NB_peaks(Sxx, freqs, sr, freqband = 200, normalization= True, slopes=(1/75,1/75))
    aci, _ = compute_ACI(Sxx, freqs, N, sr)
    ndsi = compute_NDSI(wavforme,sr,windowLength = 1024, anthrophony=[100,5000], biophony=[5000,110000])
    bi = bioacousticsIndex(Sxx, freqs, frange=(5000, 110000), R_compatible = False)
    
    EAS,_,ECV,EPS,_,_ = spectral_entropy(Sxx, freqs, frange=(5000,110000))


    indicateur = {'dB' : dB, 'ndsi': ndsi, 'aci': aci, 
                    'nbpeaks': nbpeaks, 'BI' : bi, 'EAS' : EAS, 
                    'ECV' : ECV, 'EPS' : EPS}
    
    return indicateur

import argparse

parser = argparse.ArgumentParser(description='Script to test ecoacoustic indices parameters')

parser.add_argument('--site', default=None, type=str, help='Which site to process')
parser.add_argument('--data_path', default='/bigdisk2/silentcities/', type=str, help='Path to save meta data')
parser.add_argument('--save_path', default='/bigdisk2/meta_silentcities/tests_eco', type=str, help='Path to save meta data')
parser.add_argument('--db', default=23, type=int, help='Reference dB for ACT and EVN')

args = parser.parse_args()


if __name__ == '__main__':
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
   
    NUM_CORE = multiprocessing.cpu_count() - 2
    print(f'core numbers {NUM_CORE}')
    path_audio_folder = args.data_path
    ref_dB = args.db
    site= args.site
    savepath = args.save_path
    pkl_save = os.path.join(savepath,f'tagging_site_{site}.pkl')

    meta_filename = os.path.join(savepath,f'metadata_{site}.pkl')
    Fmin, Fmax = 100,20000


    wav_files = []
    for root, dirs, files in os.walk(path_audio_folder, topdown=False):
        for name in files:
            if name[-3:].casefold() == 'wav' and name[:2] != '._':
                wav_files.append(os.path.join(root,name))

    if os.path.isfile(meta_filename):
        print(f"Loading file {meta_filename}")
        meta_file = pd.read_pickle(meta_filename)
    else:
        print('Reading metadata (listing all wave files) ')
        meta_file = metadata_generator(path_audio_folder)
        meta_file.to_pickle(meta_filename)
    print('metadata')
    meta_file = metadata_generator(path_audio_folder)
    figfile = os.path.join(savepath,f'site_{site}.html')
    print(meta_file)

    print('Dataloader')
    set_ = get_dataloader_site(path_audio_folder, meta_file, Fmin, Fmax, batch_size=NUM_CORE)

    print('processing')

    df_site = {'name':[],'start':[], 'datetime': [], 'dB':[], 'ndsi': [], 'aci': [], 'nbpeaks': [] , 'BI' : [], 'EAS':[], 'ECV' : [], 'EPS' : [],'clipwise_output' : [],'sorted_indexes' : [],'embedding' : [],}
    for batch_idx, (inputs,info) in enumerate(tqdm(set_)):

        with torch.no_grad():
            clipwise_output, labels, sorted_indexes, embedding = audio_tagging(inputs, checkpoint_path , usecuda=False)


        for idx, date_ in enumerate(info['date']):
            df_site['datetime'].append(str(date_)) 
            df_site['name'].append(str(info['name'][idx]))
            df_site['start'].append(float(info['start'][idx]))
            df_site['clipwise_output'].append(clipwise_output[idx])
            df_site['sorted_indexes'].append(sorted_indexes[idx])
            df_site['embedding'].append(embedding[idx])
            
            

    save_obj(df_site, pkl_save)