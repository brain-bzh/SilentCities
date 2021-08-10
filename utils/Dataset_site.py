import os
import multiprocessing
import datetime

import numpy as np
import scipy.signal as sig

from scipy.signal import resample
from tqdm import tqdm

import librosa
import pandas as pd


import torch

from torch.utils.data import Dataset

from utils.parameters import len_audio_s
from utils.ecoacoustics import compute_NDSI, compute_NB_peaks, compute_ACI, compute_spectrogram
from utils.alpha_indices import acoustic_events, acoustic_activity, spectral_entropy, bioacousticsIndex
# from audio_processing import DATABASE
from scipy.signal import butter, filtfilt


NUM_CORE = multiprocessing.cpu_count() - 4
print(f'core numbers {NUM_CORE}')

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_PATH = "dataloaders/"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"{DATA_PATH} folder created.")





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


def compute_ecoacoustics(wavforme, sr, ref_mindb):
    #Full band
    Sxx, freqs = compute_spectrogram(wavforme, sr)
    Sxx_dB = 10*np.log10(Sxx)
    N = len(wavforme)
    dB = 20*np.log10(np.std(wavforme))
    
    nbpeaks = compute_NB_peaks(Sxx, freqs, sr, slopes=(1/75,1/75))
    aci, _ = compute_ACI(Sxx, freqs, N, sr)
    ndsi = compute_NDSI(wavforme,sr)
    bi = bioacousticsIndex(Sxx, freqs)
    _, _, EVN,_  = acoustic_events(Sxx_dB, 1/(freqs[2]-freqs[1]), dB_threshold = ref_mindb[0]+6)
    
    _, _, ACT = acoustic_activity(Sxx_dB, dB_threshold = ref_mindb[0]+6)
    EAS,_,ECV,EPS,_,_ = spectral_entropy(Sxx, freqs)

    # filt
    wavforme = butter_bandpass_filter(wavforme, 5000, 20000, fs = sr, order=9)

    Sxx, freqs = compute_spectrogram(wavforme, sr)
    Sxx_dB = 10*np.log10(Sxx)
    N = len(wavforme)
    dB_filt = 20*np.log10(np.std(wavforme))
    nbpeaks_filt = compute_NB_peaks(Sxx, freqs, sr, slopes=(1/75,1/75))
    aci_filt, _ = compute_ACI(Sxx, freqs, N, sr)
    ndsi_filt = compute_NDSI(wavforme,sr)
    bi_filt = bioacousticsIndex(Sxx, freqs)
    _, _, EVN_filt,_  = acoustic_events(Sxx_dB, 1/(freqs[2]-freqs[1]), dB_threshold=ref_mindb[1]+6)
    
    _, _, ACT_filt = acoustic_activity(Sxx_dB, dB_threshold = ref_mindb[1]+6)
    EAS_filt,_,ECV_filt,EPS_filt,_,_ = spectral_entropy(Sxx, freqs)

    indicateur = {'dB' : dB, 'ndsi': ndsi, 'aci': aci, 
                    'nbpeaks': nbpeaks, 'BI' : bi, 'EVN' : sum(EVN), 
                    'ACT' : sum(ACT), 'EAS' : EAS, 
                    'ECV' : ECV, 'EPS' : EPS,'dB_filt' : dB_filt, 'ndsi_filt': ndsi_filt, 'aci_filt': aci_filt, 
                    'nbpeaks_filt': nbpeaks_filt, 'BI_filt' : bi_filt, 'EVN_filt' : sum(EVN_filt), 
                    'ACT_filt' : sum(ACT_filt), 'EAS_filt' : EAS_filt, 
                    'ECV_filt' : ECV_filt, 'EPS_filt' : EPS_filt}
    

    return indicateur





class Silent_dataset(Dataset):
    def __init__(self, meta_dataloader, sr, file_refdB):
        ref, sr_ = librosa.load(file_refdB, sr=None, mono=True)
        ref_filt = butter_bandpass_filter(ref, 5000, 20000, fs = sr_, order=9)
        self.ref_dB = (20*np.log10(np.std(ref)), 20*np.log10(np.std(ref_filt)))                  
        self.sr = sr
        self.meta = meta_dataloader

    def __getitem__(self, idx):
        filename = self.meta['filename'][idx]

        wav, sr = librosa.load(filename, sr=None, mono=True,
                              offset=self.meta['start'][idx], duration=len_audio_s)
        
        ecoac = compute_ecoacoustics(wav, sr, ref_mindb=self.ref_dB)

        if sr != self.sr:
            wav = resample(wav, int(len_audio_s*self.sr))
        
        # ecoac = compute_ecoacoustics(wav, self.sr, ref_mindb=self.ref_dB)
        wav = torch.tensor(wav)

        return (wav.view(int(len_audio_s*self.sr)), {'name': os.path.basename(filename), 'start': self.meta['start'][idx],
                                                        'date': self.meta['date'][idx].strftime('%Y%m%d_%H%M%S'), 
                                                        'ecoac': ecoac })

    def __len__(self):
        return len(self.meta['filename'])


def get_dataloader_site(site_ID, path_wavfile, meta_site, df_site,meta_path, database ,sample_rate=32000, batch_size=6):
    partIDidx = database[database.partID == int(site_ID)].index[0]
    file_refdB = database['ref_file'][partIDidx]
    if os.path.exists(os.path.join(meta_path, site_ID+'_metaloader.pkl')):
        meta_dataloader = pd.read_pickle(os.path.join(meta_path, site_ID+'_metaloader.pkl'))
        N = len(df_site['datetime'])
        meta_dataloader = meta_dataloader[N:]
        for root, dirs, files in os.walk(path_wavfile, topdown=False):
            for name in files:
                if name[-3:].casefold() == 'wav' and name[:2] != '._':
                    if name == file_refdB:
                        print('coucou')
                        file_refdB = os.path.join(root,name)
        
    else:
        meta_dataloader = pd.DataFrame(
            columns=['filename', 'sr', 'start', 'stop'])

        filelist = []
        for root, dirs, files in os.walk(path_wavfile, topdown=False):
            for name in files:
                if name[-3:].casefold() == 'wav' and name[:2] != '._':
                    filelist.append(os.path.join(root, name))
        
        filelist_base = [os.path.basename(file_) for file_ in filelist]

        for idx, wavfile in tqdm(enumerate(meta_site['filename'])):

            len_file = meta_site['length'][idx]
            sr_in = meta_site['sr'][idx]

            duration = len_file/sr_in

            nb_win = int(duration // len_audio_s )

            for win in range(nb_win):
                delta = datetime.timedelta(seconds=int((win*len_audio_s)))
                meta_dataloader = meta_dataloader.append({'filename': filelist[filelist_base.index(wavfile)], 'sr': sr_in, 'start': (
                    win*len_audio_s), 'stop': ((win+1)*len_audio_s), 'len': len_file, 'date': meta_site['datetime'][idx] + delta}, ignore_index=True)
            # if duration % len_audio_s == float(0):
            #     delta = datetime.timedelta(seconds=int((nb_win-1)*len_audio_s))
            #     meta_dataloader = meta_dataloader.append({'filename': filelist[filelist_base.index(wavfile)], 'sr': sr_in, 'start': (
            #         duration - len_audio_s), 'stop': (duration), 'len': len_file, 'date': meta_site['datetime'][idx] + delta}, ignore_index=True)
            if wavfile == file_refdB:
                file_refdB = filelist[filelist_base.index(wavfile)]
            
        meta_dataloader.to_pickle(os.path.join(meta_path, site_ID+'_metaloader.pkl'))
    print(meta_dataloader)

    site_set = Silent_dataset(meta_dataloader.reset_index(drop=True), sample_rate, file_refdB)
    site_set = torch.utils.data.DataLoader(
        site_set, batch_size=batch_size, shuffle=False, num_workers=NUM_CORE-1)

    return site_set




if __name__ == '__main__':

    site_set = get_dataloader_site('0001', '/media/nicolas/Silent/0001')
    for batch_idx, (inputs, info) in enumerate(site_set):
        print(inputs)
        print(info)