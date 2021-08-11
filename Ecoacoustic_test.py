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
from utils import OctaveBand


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
    # refdB = [meta_site.iloc[0]['dB'], meta_site.iloc[0]['filename']]

    filelist = []
    for root, dirs, files in os.walk(path_wavfile, topdown=False):
        for name in files:
            if name[-3:].casefold() == 'wav' and name[:2] != '._':
                filelist.append(os.path.join(root, name))
    filelist_base = [os.path.basename(file_) for file_ in filelist]
    refdB = [meta_site.iloc[0]['dB'], filelist[filelist_base.index(meta_site.iloc[0]['filename'])]]
    for idx, wavfile in enumerate(tqdm(meta_site['filename'])):

        len_file = meta_site['length'][idx]
        sr_in = meta_site['sr'][idx]
        curdB = meta_site['dB'][idx]
        if curdB < refdB[0]:
            refdB = [curdB, filelist[filelist_base.index(wavfile)]]
        duration = len_file / sr_in
        nb_win = int(duration // len_audio_s)

        for win in range(nb_win):
            delta = datetime.timedelta(seconds=int((win * len_audio_s)))
            meta_dataloader = meta_dataloader.append({'filename': filelist[filelist_base.index(wavfile)], 'sr': sr_in, 'start': (
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

    for idx, wavfile in enumerate(tqdm(filelist)):
        _, meta = utils.read_audio_hdr(wavfile, False)  # meta data

        x, sr = librosa.load(wavfile, sr=None, mono=True)
        Df = Df.append({'datetime': meta['datetime'], 'filename': os.path.basename(wavfile), 'length': len(x), 'sr': sr,
                        'dB': 20 * np.log10(np.std(x))}, ignore_index=True)

    Df = Df.sort_values('datetime')

    return Df.reset_index(drop=True)



def compute_ecoacoustics(wavforme, sr, Fmin, Fmax, refdB):


    wavforme = butter_bandpass_filter(wavforme, Fmin, Fmax, fs=sr)
    dB_band, freq = OctaveBand.octavefilter(wavforme, fs=sr, fraction=1, order=4, limits=[100, 20000], show=0)
    # dB_band = [0]*8
    Sxx, freqs = compute_spectrogram(wavforme, sr)

    Sxx_dB = 10 * np.log10(Sxx)
    N = len(wavforme)
    dB = 20 * np.log10(np.std(wavforme))

    # nbpeaks = compute_NB_peaks(Sxx, freqs, sr, freqband=200, normalization=True, slopes=(1 / 75, 1 / 75))
    # wide : 2000 - 20000 : narrow : 5000 : 15000
    min_anthro_bin = np.argmin([abs(e - 5000) for e in freqs])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin = np.argmin([abs(e - 15000) for e in freqs])  # max freq of anthrophony in samples (or bin)
    aci_N, _ = compute_ACI(Sxx[min_anthro_bin:max_anthro_bin,:], freqs[min_anthro_bin:max_anthro_bin], 100, sr) # Filtrage 2000 : 20000 (biophony)
    ndsi_N = compute_NDSI(wavforme, sr, windowLength=1024, anthrophony=[1000, 5000], biophony=[5000, 15000])
    bi_N = bioacousticsIndex(Sxx, freqs, frange=(5000, 15000), R_compatible=False)

    min_anthro_bin = np.argmin([abs(e - 2000) for e in freqs])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin = np.argmin([abs(e - 20000) for e in freqs])  # max freq of anthrophony in samples (or bin)
    aci_W, _ = compute_ACI(Sxx[min_anthro_bin:max_anthro_bin, :], freqs, 100, sr)  # Filtrage 2000 : 20000 (biophony)
    ndsi_W = compute_NDSI(wavforme, sr, windowLength=1024, anthrophony=[1000, 2000], biophony=[2000, 20000])
    bi_W = bioacousticsIndex(Sxx, freqs, frange=(2000, 20000), R_compatible=False)

    # wide 1000 - 20000 narrow 5000 - 15000


    EAS_N, _, ECV_N, EPS_N, _, _ = spectral_entropy(Sxx, freqs, frange=(5000, 15000))
    EAS_W, _, ECV_W, EPS_W, _, _ = spectral_entropy(Sxx, freqs, frange=(2000, 20000))


    ### specific code to calculate ACT and EVN with many ref Db offset

    _, ACT, _ = acoustic_activity(10*np.log10(np.abs(wavforme)**2), dB_threshold=refdB + 12, axis=-1)
    ACT = np.sum(np.asarray(ACT))/sr

    indicateur = {'dB': dB, 'ndsi_N': ndsi_N, 'aci_N': aci_N,
                  'BI_N': bi_N, 'EAS_N': EAS_N,
                  'ECV_N': ECV_N, 'EPS_N': EPS_N,'ndsi_W': ndsi_W, 'aci_W': aci_W,
                  'BI_W': bi_W, 'EAS_W': EAS_W,
                  'ECV_W': ECV_W, 'EPS_W': EPS_W, 'ACT':ACT,
                  'POWERB_126':dB_band[0], 'POWERB_251':dB_band[1], 'POWERB_501':dB_band[2], 'POWERB_1k':dB_band[3], 'POWERB_2k':dB_band[4], 'POWERB_4k':dB_band[5], 'POWERB_8k':dB_band[6], 'POWERB_16k':dB_band[7]}

    # indicateur[f"EVN_ref+{cur_offset}"] = np.sum(np.asarray(EVN))

    return indicateur


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test ecoacoustic indices parameters')

    parser.add_argument('--site', default=1, type=int, help='Which site to process')
    parser.add_argument('--nfiles', default=1000, type=int,
                        help='How many files per site (will take the first nfiles files)')
    parser.add_argument('--data_path', default='fortesteco', type=str, help='Path to save meta data')
    parser.add_argument('--save_path', default='/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/NEW3', type=str,
                        help='Path to save meta data')
    parser.add_argument('--reject_core', default=1,
                        type=int,
                        help='number of rejected core during the multiprocess calculation')

    args = parser.parse_args()

    NUM_CORE = multiprocessing.cpu_count() - args.reject_core
    print(f'core numbers {NUM_CORE}')
    site = args.site
    # for site in tqdm([4, 11, 25, 36, 38, 48, 52, 61, 62, 77, 87, 115, 120, 121, 132, 153, 158, 159, 190, 229, 234, 276, 292, 346, 371, 388]):
    site = f"{site:04d}"
    nfiles = args.nfiles
    if nfiles == 0:
        nfiles = None
    path_audio_folder = os.path.join(args.data_path, site)

    savepath = args.save_path
    CSV_SAVE = os.path.join(savepath, f'site_{site}.csv')
    figfile = os.path.join(savepath, f'site_{site}.html')
    meta_filename = os.path.join(savepath, f'{site}.pkl')
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
        _meta_file = pd.read_pickle(meta_filename)
        _meta_file = _meta_file.sort_values('datetime')
        _meta_file = _meta_file.reset_index(drop=True)
        meta_file = _meta_file[:nfiles]

    else:
        print('Reading metadata (listing all wave files) ')
        _meta_file = metadata_generator(path_audio_folder, nfiles=nfiles)
        _meta_file.to_pickle(meta_filename)
        meta_file = _meta_file[: nfiles]

    print('Preparing Dataloader (which also means calculating all indices)')
    set_ = get_dataloader_site(path_audio_folder, meta_file, Fmin, Fmax, batch_size=int(NUM_CORE*2))

    df_site = {'name': [], 'start': [], 'datetime': [], 'dB': [], 'ndsi_N': [], 'aci_N': [],
                    'BI_N': [], 'EAS_N': [],
                    'ECV_N': [], 'EPS_N': [] ,'ndsi_W': [], 'aci_W': [],
                    'BI_W': [], 'EAS_W': [],
                    'ECV_W': [], 'EPS_W': [], 'ACT':[],
                'POWERB_126':[], 'POWERB_251':[], 'POWERB_501':[], 'POWERB_1k':[], 'POWERB_2k':[], 'POWERB_4k':[], 'POWERB_8k':[], 'POWERB_16k':[]}


    for batch_idx, info in enumerate(set_):
        for idx, date_ in enumerate(info['date']):
            df_site['datetime'].append(str(date_))
            df_site['name'].append(str(info['name'][idx]))
            df_site['start'].append(float(info['start'][idx]))
            for key in info['ecoac'].keys():
                df_site[key].append(float(info['ecoac'][key].numpy()[idx]))

    df_site = pd.DataFrame(df_site)
    df_site.to_csv(CSV_SAVE, index=False)
    df_site = df_site.sort_values('datetime').reset_index(drop = True)
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
