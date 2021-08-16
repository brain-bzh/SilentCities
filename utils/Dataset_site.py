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
from utils import OctaveBand


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



def compute_ecoacoustics(wavforme, sr, Fmin, Fmax, refdB):


    wavforme = butter_bandpass_filter(wavforme, Fmin, Fmax, fs=sr)
    dB_band, _ = OctaveBand.octavefilter(wavforme, fs=sr, fraction=1, order=4, limits=[100, 20000], show=0)
    # dB_band = [0]*8
    Sxx, freqs = compute_spectrogram(wavforme, sr)

    # Sxx_dB = 10 * np.log10(Sxx)
    # N = len(wavforme)
    dB = 20 * np.log10(np.std(wavforme))

    # nbpeaks = compute_NB_peaks(Sxx, freqs, sr, freqband=200, normalization=True, slopes=(1 / 75, 1 / 75))
    # wide : 2000 - 20000 : narrow : 5000 : 15000
    min_anthro_bin = np.argmin([abs(e - 5000) for e in freqs])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin = np.argmin([abs(e - 15000) for e in freqs])  # max freq of anthrophony in samples (or bin)
    aci_N, _ = compute_ACI(Sxx[min_anthro_bin:max_anthro_bin,:], freqs[min_anthro_bin:max_anthro_bin], 100, sr) 
    ndsi_N = compute_NDSI(wavforme, sr, windowLength=1024, anthrophony=[1000, 5000], biophony=[5000, 15000])
    bi_N = bioacousticsIndex(Sxx, freqs, frange=(5000, 15000), R_compatible=False)

    min_anthro_bin = np.argmin([abs(e - 2000) for e in freqs])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin = np.argmin([abs(e - 20000) for e in freqs])  # max freq of anthrophony in samples (or bin)
    aci_W, _ = compute_ACI(Sxx[min_anthro_bin:max_anthro_bin, :], freqs, 100, sr)  
    ndsi_W = compute_NDSI(wavforme, sr, windowLength=1024, anthrophony=[1000, 2000], biophony=[2000, 20000])
    bi_W = bioacousticsIndex(Sxx, freqs, frange=(2000, 20000), R_compatible=False)

    # wide 1000 - 20000 narrow 5000 - 15000


    EAS_N, _, ECV_N, EPS_N, _, _ = spectral_entropy(Sxx, freqs, frange=(5000, 15000))
    EAS_W, _, ECV_W, EPS_W, _, _ = spectral_entropy(Sxx, freqs, frange=(2000, 20000))


    ### specific code to calculate ACT and EVN with many ref Db offset

    _, ACT, _ = acoustic_activity(10*np.log10(np.abs(wavforme)**2), dB_threshold=refdB[1] + 12, axis=-1)
    ACT = np.sum(np.asarray(ACT))/sr

    indicateur = {'dB': dB, 'ndsi_N': ndsi_N, 'aci_N': aci_N,
                  'BI_N': bi_N, 'EAS_N': EAS_N,
                  'ECV_N': ECV_N, 'EPS_N': EPS_N,'ndsi_W': ndsi_W, 'aci_W': aci_W,
                  'BI_W': bi_W, 'EAS_W': EAS_W,
                  'ECV_W': ECV_W, 'EPS_W': EPS_W, 'ACT':ACT,
                  'POWERB_126':dB_band[0], 'POWERB_251':dB_band[1], 'POWERB_501':dB_band[2], 'POWERB_1k':dB_band[3], 'POWERB_2k':dB_band[4], 'POWERB_4k':dB_band[5], 'POWERB_8k':dB_band[6], 'POWERB_16k':dB_band[7]}

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
        
        ecoac = compute_ecoacoustics(wav, sr, Fmin = 100, Fmax=20000, ref_mindb=self.ref_dB)

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