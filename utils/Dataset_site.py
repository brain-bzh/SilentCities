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
from utils.ecoacoustics import compute_NDSI, compute_NB_peaks, compute_ACI
# from audio_processing import DATABASE



NUM_CORE = multiprocessing.cpu_count()
print(f'core numbers {NUM_CORE}')

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

DATA_PATH = "dataloaders/"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"{DATA_PATH} folder created.")


class Silent_dataset(Dataset):
    def __init__(self, meta_dataloader, sr):
        self.sr = sr
        self.meta = meta_dataloader

    def __getitem__(self, idx):
        filename = self.meta['filename'][idx]

        wav, _ = librosa.load(filename, sr=None, mono=True,
                              offset=self.meta['start'][idx], duration=len_audio_s)

        if int(self.meta['sr'][idx]) != self.sr:
            wav = resample(wav, int(len_audio_s*self.sr))

        ndsi = compute_NDSI(wav, self.sr)
        nbpeaks = compute_NB_peaks(wav, self.sr)
        aci, _ = compute_ACI(wav, self.sr)

        wav = torch.tensor(wav)

        return (wav.view(int(len_audio_s*self.sr)), {'name': os.path.basename(filename), 
                                                        'date': self.meta['date'][idx].strftime('%Y%m%d_%H%M%S'), 
                                                        'ecoac': {'ndsi': ndsi, 'aci': aci, 'nbpeaks': nbpeaks}})

    def __len__(self):
        return len(self.meta['filename'])


def get_dataloader_site(site_ID, path_wavfile, meta_site, meta_path ,sample_rate=32000, batch_size=6):
    
    if os.path.exists(os.path.join(meta_path, site_ID+'_metaloader.pkl')):
        meta_dataloader = pd.read_pickle(os.path.join(meta_path, site_ID+'_metaloader.pkl'))
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

            nb_win = int(duration//len_audio_s)

            for win in range(nb_win):
                delta = datetime.timedelta(seconds=int((win*len_audio_s)))
                meta_dataloader = meta_dataloader.append({'filename': filelist[filelist_base.index(wavfile)], 'sr': sr_in, 'start': (
                    win*len_audio_s), 'stop': ((win+1)*len_audio_s), 'len': len_file, 'date': meta_site['datetime'][idx] + delta}, ignore_index=True)
        meta_dataloader.to_pickle(os.path.join(meta_path, site_ID+'_metaloader.pkl'))
    print(meta_dataloader)

    site_set = Silent_dataset(meta_dataloader, sample_rate)
    site_set = torch.utils.data.DataLoader(
        site_set, batch_size=batch_size, shuffle=False, num_workers=NUM_CORE-1)

    return site_set




if __name__ == '__main__':

    site_set = get_dataloader_site('0001', '/media/nicolas/Silent/0001')
    for batch_idx, (inputs, info) in enumerate(site_set):
        print(inputs)
        print(info)