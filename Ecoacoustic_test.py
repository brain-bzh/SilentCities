import os
import multiprocessing
from tqdm import tqdm
import datetime

import librosa
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal as sig

import torch
from torch.utils.data import Dataset

from utils import utils
from utils.ecoacoustics import compute_NDSI, compute_NB_peaks, compute_ACI, compute_spectrogram
from utils.alpha_indices import acoustic_events, acoustic_activity, spectral_entropy, bioacousticsIndex
from scipy.signal import butter, filtfilt
from utils.parameters import len_audio_s


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

        
        ecoac = compute_ecoacoustics(wav,sr, self.ref_dB,self.Fmin, self.Fmax)

        return {'name': os.path.basename(filename), 'start': self.meta['start'][idx],
                                                        'date': self.meta['date'][idx].strftime('%Y%m%d_%H%M%S'), 
                                                        'ecoac': ecoac }
    def __len__(self):
        return len(self.meta['filename'])
    
def get_dataloader_site(path_wavfile, meta_site,Fmin, Fmax, batch_size=1):

    meta_dataloader = pd.DataFrame(
        columns=['filename', 'sr', 'start', 'stop'])

    for idx, wavfile in tqdm(enumerate(meta_site['filename'])):

        len_file = meta_site['length'][idx]
        sr_in = meta_site['sr'][idx]

        duration = len_file/sr_in
        nb_win = int(duration // len_audio_s )

        for win in range(nb_win):
            delta = datetime.timedelta(seconds=int((win*len_audio_s)))
            meta_dataloader = meta_dataloader.append({'filename': wavfile, 'sr': sr_in, 'start': (
                win*len_audio_s), 'stop': ((win+1)*len_audio_s), 'len': len_file, 'date': meta_site['datetime'][idx] + delta}, ignore_index=True)
        if duration % len_audio_s == float(0):
            delta = datetime.timedelta(seconds=int((nb_win-1)*len_audio_s))
            meta_dataloader = meta_dataloader.append({'filename':wavfile, 'sr': sr_in, 'start': (
                duration - len_audio_s), 'stop': (duration), 'len': len_file, 'date': meta_site['datetime'][idx] + delta}, ignore_index=True)

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
        
    for idx, wavfile in tqdm(enumerate(filelist)):
        _, meta = utils.read_audio_hdr(wavfile, False) #meta data

        if idx == 0 : sr, x = wav.read(wavfile)
        Df = Df.append({'datetime': meta['datetime'], 'filename': wavfile, 'length' : len(x), 'sr' : sr}, ignore_index=True)
            
    Df = Df.sort_values('datetime')
    return(Df)



def compute_ecoacoustics(wavforme,sr, ref_mindb, Fmin, Fmax):

    ref_mindb -=90 #int16 to -1 1
    wavforme = butter_bandpass_filter(wavforme, Fmin, Fmax, fs = sr)

    Sxx, freqs = compute_spectrogram(wavforme, sr)

    Sxx_dB = 10*np.log10(Sxx)
    N = len(wavforme)
    dB = 20*np.log10(np.std(wavforme))
    
    nbpeaks = compute_NB_peaks(Sxx, freqs, sr, freqband = 200, normalization= True, slopes=(1/75,1/75))
    aci, _ = compute_ACI(Sxx, freqs, N, sr)
    ndsi = compute_NDSI(wavforme,sr,windowLength = 1024, anthrophony=[1000,2000], biophony=[2000,11000])
    bi = bioacousticsIndex(Sxx, freqs, frange=(2000, 15000), R_compatible = False)
    _, _, EVN,_  = acoustic_events(Sxx_dB, 1/(freqs[2]-freqs[1]), dB_threshold = ref_mindb+6)
    
    _, _, ACT = acoustic_activity(Sxx_dB, dB_threshold = ref_mindb+6)
    EAS,_,ECV,EPS,_,_ = spectral_entropy(Sxx, freqs, frange=None)


    indicateur = {'dB' : dB, 'ndsi': ndsi, 'aci': aci, 
                    'nbpeaks': nbpeaks, 'BI' : bi, 'EVN' : sum(EVN), 
                    'ACT' : sum(ACT), 'EAS' : EAS, 
                    'ECV' : ECV, 'EPS' : EPS}
    
    return indicateur

if __name__ == '__main__':
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
   
    NUM_CORE = multiprocessing.cpu_count()
    print(f'core numbers {NUM_CORE}')
    path_audio_folder = '/Volumes/LaCie/0292/TEST'
    CSV_SAVE = 'save.csv'
    ref_dB = 23
    Fmin, Fmax = 100,20000


    wav_files = []
    for root, dirs, files in os.walk(path_audio_folder, topdown=False):
        for name in files:
            if name[-3:].casefold() == 'wav' and name[:2] != '._':
                wav_files.append(os.path.join(root,name))


    print('metadata')
    meta_file = metadata_generator(path_audio_folder)
    print(meta_file)

    print('Dataloader')
    set_ = get_dataloader_site(path_audio_folder, meta_file, Fmin, Fmax, batch_size=8)

    print('processing')

    df_site = {'name':[],'start':[], 'datetime': [], 'dB':[], 'ndsi': [], 'aci': [], 'nbpeaks': [] , 'BI' : [], 'EVN' : [], 'ACT' : [], 'EAS':[], 'ECV' : [], 'EPS' : []}
    for batch_idx, info in tqdm(enumerate(set_)):
        for idx, date_ in enumerate(info['date']):
            df_site['datetime'].append(str(date_)) 
            df_site['name'].append(str(info['name'][idx]))
            df_site['start'].append(float(info['start'][idx]))
            for key in info['ecoac'].keys():
                df_site[key].append(float(info['ecoac'][key].numpy()[idx])) 
    df_site = pd.DataFrame(df_site)
    df_site.to_csv(CSV_SAVE)

    for idx, k in enumerate(df_site['datetime']) :
        df_site['datetime'][idx] = datetime.datetime.strptime(k, '%Y%m%d_%H%M%S')


    indic = ['dB', 'ndsi', 'aci', 'nbpeaks', 'BI', 'EVN', 'ACT', 'EAS', 'ECV', 'EPS']

    fig = make_subplots(rows=5, cols=2,print_grid=True, subplot_titles=indic,shared_xaxes='all')
    for idx, k in enumerate(indic): 
        fig.add_trace(go.Scatter(x=df_site['datetime'], y=df_site[k]),
                row=(idx//2)+1, col=(idx%2)+1)

    fig.update(layout_showlegend=False)
    fig.show()
    print(df_site)
