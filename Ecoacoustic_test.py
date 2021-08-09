import os
import multiprocessing
from tqdm import tqdm
import datetime

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import librosa
import pandas as pd
import numpy as np
# import scipy.io.wavfile as wav
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
    def __init__(self, meta_dataloader, Fmin, Fmax, refdB):
        self.meta = meta_dataloader
        self.Fmin, self.Fmax = Fmin, Fmax

        dBref, sr = librosa.load(refdB[1], sr=None, mono=True)
        dBref = butter_bandpass_filter(dBref, Fmin, Fmax, fs=sr)
        print('---------------'*10)
        print(refdB[0], 20*np.log10(np.std(dBref)))
        print('---------------' * 10)

        self.refdB = 20*np.log10(np.std(dBref))

    def __getitem__(self, idx):
        filename = self.meta['filename'][idx]

        wav, sr = librosa.load(filename, sr=None, mono=True,
                               offset=self.meta['start'][idx], duration=len_audio_s)

        ecoac = compute_ecoacoustics(wav, sr, self.Fmin, self.Fmax, self.refdB)

        return {'name': os.path.basename(filename), 'start': self.meta['start'][idx],
                'date': self.meta['date'][idx].strftime('%Y%m%d_%H%M%S'),
                'ecoac': ecoac}

    def __len__(self):
        return len(self.meta['filename'])


def get_dataloader_site(path_wavfile, meta_site, Fmin, Fmax, batch_size=1):
    meta_dataloader = pd.DataFrame(
        columns=['filename', 'sr', 'start', 'stop'])
    refdB = [meta_site.iloc[0]['dB'], meta_site.iloc[0]['filename']]
    for idx, wavfile in enumerate(tqdm(meta_site['filename'])):

        len_file = meta_site['length'][idx]
        sr_in = meta_site['sr'][idx]
        curdB = meta_site['dB'][idx]
        if curdB < refdB[0]:
            refdB = [curdB, wavfile]
        duration = len_file / sr_in
        nb_win = int(duration // len_audio_s)

        for win in range(nb_win):
            delta = datetime.timedelta(seconds=int((win * len_audio_s)))
            meta_dataloader = meta_dataloader.append({'filename': wavfile, 'sr': sr_in, 'start': (
                    win * len_audio_s), 'stop': ((win + 1) * len_audio_s), 'len': len_file,
                                                      'date': meta_site['datetime'][idx] + delta}, ignore_index=True)
        # if duration % len_audio_s == float(0):
        #     delta = datetime.timedelta(seconds=int((nb_win-1)*len_audio_s))
        #     meta_dataloader = meta_dataloader.append({'filename': filelist[filelist_base.index(wavfile)], 'sr': sr_in, 'start': (
        #         duration - len_audio_s), 'stop': (duration), 'len': len_file, 'date': meta_site['datetime'][idx] + delta}, ignore_index=True)

    site_set = Silent_dataset(meta_dataloader.reset_index(drop=True), Fmin, Fmax, refdB)
    site_set = torch.utils.data.DataLoader(
        site_set, batch_size=batch_size, shuffle=False, num_workers=NUM_CORE)

    return site_set


def metadata_generator(folder, nfiles=None):
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
    if nfiles is not None:
        filelist = filelist[:nfiles]
    for idx, wavfile in enumerate(tqdm(filelist)):
        _, meta = utils.read_audio_hdr(wavfile, False)  # meta data

        x, sr = librosa.load(wavfile, sr=None, mono=True)
        Df = Df.append({'datetime': meta['datetime'], 'filename': wavfile, 'length': len(x), 'sr': sr,
                        'dB': 20 * np.log10(np.std(x))}, ignore_index=True)

    Df = Df.sort_values('datetime')
    return (Df)


def compute_ecoacoustics(wavforme, sr, Fmin, Fmax, refdB):
    wavforme = butter_bandpass_filter(wavforme, Fmin, Fmax, fs=sr)

    Sxx, freqs = compute_spectrogram(wavforme, sr)

    Sxx_dB = 10 * np.log10(Sxx)
    N = len(wavforme)
    dB = 20 * np.log10(np.std(wavforme))

    nbpeaks = compute_NB_peaks(Sxx, freqs, sr, freqband=200, normalization=True, slopes=(1 / 75, 1 / 75))
    aci, _ = compute_ACI(Sxx, freqs, N, sr)
    ndsi = compute_NDSI(wavforme, sr, windowLength=1024, anthrophony=[1000, 5000], biophony=[5000, 20000])
    bi = bioacousticsIndex(Sxx, freqs, frange=(5000, 20000), R_compatible=False)

    EAS, _, ECV, EPS, _, _ = spectral_entropy(Sxx, freqs, frange=(1000, 10000))

    indicateur = {'dB': dB, 'ndsi': ndsi, 'aci': aci,
                  'nbpeaks': nbpeaks, 'BI': bi, 'EAS': EAS,
                  'ECV': ECV, 'EPS': EPS}

    ### specific code to calculate ACT and EVN with many ref Db offset

    list_offset = [5, 10, 15, 20, 25, 30, 35]
    for cur_offset in list_offset:
        _, _, EVN, _ = acoustic_events(Sxx_dB, 1 / (freqs[2] - freqs[1]), dB_threshold=refdB + cur_offset, rejectDuration = sr/100)
        _, ACT, _ = acoustic_activity(10*np.log10(np.abs(sig.hilbert(wavforme))**2), dB_threshold=refdB + cur_offset, axis=-1)

        indicateur[f"ACT_ref+{cur_offset}"] = ACT/N
        indicateur[f"EVN_ref+{cur_offset}"] = sum(EVN)

    return indicateur


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test ecoacoustic indices parameters')

    parser.add_argument('--site', default=None, type=str, help='Which site to process')
    parser.add_argument('--nfiles', default=1000, type=int,
                        help='How many files per site (will take the first nfiles files)')
    parser.add_argument('--data_path', default='/Volumes/NICOLAS', type=str, help='Path to save meta data')
    parser.add_argument('--save_path', default='/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique', type=str,
                        help='Path to save meta data')

    args = parser.parse_args()

    NUM_CORE = multiprocessing.cpu_count() - 2
    print(f'core numbers {NUM_CORE}')
    site = args.site
    nfiles = args.nfiles
    if nfiles == 0:
        nfiles = None
    path_audio_folder = os.path.join(args.data_path, site)
    print(path_audio_folder)

    savepath = args.save_path
    CSV_SAVE = os.path.join(savepath, f'site_{site}.csv')
    figfile = os.path.join(savepath, f'site_{site}.html')
    meta_filename = os.path.join(savepath, f'metadata_{site}.pkl')
    # print(CSV_SAVE)
    Fmin, Fmax = 100, 20000

    """wav_files = []
    for root, dirs, files in os.walk(path_audio_folder, topdown=False):
        if len(wav_files) < nfiles:
                
            for name in files:
                if name[-3:].casefold() == 'wav' and name[:2] != '._':
                    wav_files.append(os.path.join(root,name))
        else:
            break"""

    if os.path.isfile(meta_filename):
        print(f"Loading file {meta_filename}")
        meta_file = pd.read_pickle(meta_filename)
    else:
        print('Reading metadata (listing all wave files) ')
        meta_file = metadata_generator(path_audio_folder, nfiles=nfiles)
        meta_file.to_pickle(meta_filename)

    print(meta_file)

    print('Preparing Dataloader (which also means calculating all indices)')
    set_ = get_dataloader_site(path_audio_folder, meta_file, Fmin, Fmax, batch_size=NUM_CORE)

    df_site = {'name': [], 'start': [], 'datetime': [], 'dB': [], 'ndsi': [], 'aci': [], 'nbpeaks': [], 'BI': [],
               'EAS': [], 'ECV': [], 'EPS': []}

    list_offset = [5, 10, 15, 20, 25, 30, 35]
    for offset in list_offset:
        df_site[f"ACT_ref+{offset}"] = []
        df_site[f"EVN_ref+{offset}"] = []

    for batch_idx, info in enumerate(tqdm(set_)):
        for idx, date_ in enumerate(info['date']):
            df_site['datetime'].append(str(date_))
            df_site['name'].append(str(info['name'][idx]))
            df_site['start'].append(float(info['start'][idx]))
            for key in info['ecoac'].keys():
                df_site[key].append(float(info['ecoac'][key].numpy()[idx]))
    df_site = pd.DataFrame(df_site)

    df_site = df_site.sort_values('datetime').reset_index()
    df_site.to_csv(CSV_SAVE, index=False)

    """
    for idx, k in enumerate(df_site['datetime']) :
        df_site['datetime'][idx] = datetime.datetime.strptime(k, '%Y%m%d_%H%M%S')

    print(df_site)
    
    indic = ['dB', 'ndsi', 'aci', 'nbpeaks', 'BI', 'EVN', 'ACT', 'EAS', 'ECV', 'EPS']

    fig = make_subplots(rows=5, cols=2,print_grid=True, subplot_titles=indic,shared_xaxes='all')
    for idx, k in enumerate(indic): 
        fig.add_trace(go.Scatter(x=df_site['datetime'], y=df_site[k]),
                row=(idx//2)+1, col=(idx%2)+1)

    fig.update(layout_showlegend=False)
    fig.write_html(figfile)
    """

####   dans la boucle de get_dataloader_site, rajouter une condition pour choper le minimum du dB
### le mettre  e n paramètre de la fonction compute ecoacoustics
### plus besoin du -90  ; faire varier dB_threshold = ref_mindb+variable
