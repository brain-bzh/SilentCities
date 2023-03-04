#!/usr/bin/env python

"""
from https://github.com/patriceguyot/Acoustic_Indices

    Set of functions to compute acoustic indices in the framework of Soundscape Ecology.
    Some features are inspired or ported from those proposed in:
        - seewave R package (http://rug.mnhn.fr/seewave/) / Jerome Sueur, Thierry Aubin and  Caroline Simonis
        - soundecology R package (http://cran.r-project.org/web/packages/soundecology/index.html) / Luis J. Villanueva-Rivera and Bryan C. Pijanowski
    This file use an object oriented type for audio files described in the file "acoustic_index.py".
"""

__author__ = "Patrice Guyot"
__version__ = "0.4"
__credits__ = ["Patrice Guyot", "Alice Eldridge", "Mika Peck"]
__email__ = ["guyot.patrice@gmail.com", "alicee@sussex.ac.uk", "m.r.peck@sussex.ac.uk"]
__status__ = "Development"


from scipy import signal, fftpack
import numpy as np
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_NDSI(wave,sr, windowLength = 1024, anthrophony=[1000,2000], biophony=[2000,11000]):
    """
    Compute Normalized Difference Sound Index from an audio signal.
    This function compute an estimate power spectral density using Welch's method.
    Reference: Kasten, Eric P., Stuart H. Gage, Jordan Fox, and Wooyeong Joo. 2012. The Remote Environ- mental Assessment Laboratory's Acoustic Library: An Archive for Studying Soundscape Ecology. Ecological Informatics 12: 50-67.
    windowLength: the length of the window for the Welch's method.
    anthrophony: list of two values containing the minimum and maximum frequencies (in Hertz) for antrophony.
    biophony: list of two values containing the minimum and maximum frequencies (in Hertz) for biophony.
    Inspired by the seewave R package, the soundecology R package and the original matlab code from the authors.
    """

    #frequencies, pxx = signal.welch(file.sig_float, fs=file.sr, window='hamming', nperseg=windowLength, noverlap=windowLength/2, nfft=windowLength, detrend=False, return_onesided=True, scaling='density', axis=-1) # Estimate power spectral density using Welch's method
    # TODO change of detrend for apollo
    frequencies, pxx = signal.welch(wave, fs=sr, window='hamming', nperseg=windowLength, noverlap=windowLength/2, nfft=windowLength, detrend='constant', return_onesided=True, scaling='density', axis=-1) # Estimate power spectral density using Welch's method
    avgpow = pxx * frequencies[1] # use a rectangle approximation of the integral of the signal's power spectral density (PSD)
    #avgpow = avgpow / np.linalg.norm(avgpow, ord=2) # Normalization (doesn't change the NDSI values. Slightly differ from the matlab code).

    min_anthro_bin=np.argmin([abs(e - anthrophony[0]) for e in frequencies])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin=np.argmin([abs(e - anthrophony[1]) for e in frequencies])  # max freq of anthrophony in samples (or bin)
    min_bio_bin=np.argmin([abs(e - biophony[0]) for e in frequencies])  # min freq of biophony in samples (or bin)
    max_bio_bin=np.argmin([abs(e - biophony[1]) for e in frequencies])  # max freq of biophony in samples (or bin)

    anthro = np.sum(avgpow[min_anthro_bin:max_anthro_bin])
    bio = np.sum(avgpow[min_bio_bin:max_bio_bin])

    ndsi = (bio - anthro) / (bio + anthro)
    return ndsi



#--------------------------------------------------------------------------------------------------------------------------------------------------
def compute_spectrogram(sig,sr, windowLength=512, windowHop= 256, square=True, windowType='hann', centered=False, normalized = False ):
    """
    Compute a spectrogram of an audio signal.
    Return a list of list of values as the spectrogram, and a list of frequencies.
    Keyword arguments:
    file -- the real part (default 0.0)
    Parameters:
    file: an instance of the AudioFile class.
    windowLength: length of the fft window (in samples)
    windowHop: hop size of the fft window (in samples)
    ##### scale_audio: if set as True, the signal samples are scale between -1 and 1 (as the audio convention). If false the signal samples remains Integers (as output from scipy.io.wavfile)
    square: if set as True, the spectrogram is computed as the square of the magnitude of the fft. If not, it is the magnitude of the fft.
    hamming: if set as True, the spectrogram use a correlation with a hamming window.
    centered: if set as true, each resulting fft is centered on the corresponding sliding window
    normalized: if set as true, divide all values by the maximum value
    """

    #if scale_audio: ### this is the case with our files
    #    sig = file.sig_float # use signal with float between -1 and 1
    #else:
    #    sig = file.sig_int # use signal with integers

    W = signal.get_window(windowType, windowLength, fftbins=False)
    halfWindowLength = int(windowLength/2)

    if centered:
        time_shift = int(windowLength/2)
        times = range(time_shift, len(sig)+1-time_shift, windowHop) # centered
        frames = [sig[i-time_shift:i+time_shift]*W for i in times] # centered frames
    else:
        times = range(0, len(sig)-windowLength+1, windowHop)
        frames = [sig[i:i+windowLength]*W for i in times]

    if square:
        spectro =  [abs(np.fft.rfft(frame, windowLength))[0:halfWindowLength]**2 for frame in frames]
    else:
        spectro =  [abs(np.fft.rfft(frame, windowLength))[0:halfWindowLength] for frame in frames]

    spectro=np.transpose(spectro) # set the spectro in a friendly way

    if normalized:
        spectro = spectro/np.max(spectro) # set the maximum value to 1 y

    frequencies = [e * (sr/2) / float(windowLength / 2) for e in range(halfWindowLength)] # vector of frequency<-bin in the spectrogram
    return spectro, np.array(frequencies)

def compute_NB_peaks(spectro, frequencies, sr, freqband = 200, normalization= True, slopes=(0.01,0.01)):
    """
    Counts the number of major frequency peaks obtained on a mean spectrum.
    spectro: spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    freqband: frequency threshold parameter (in Hz). If the frequency difference of two successive peaks is less than this threshold, then the peak of highest amplitude will be kept only.
    normalization: if set at True, the mean spectrum is scaled between 0 and 1
    slopes: amplitude slope parameter, a tuple of length 2. Refers to the amplitude slopes of the peak. The first value is the left slope and the second value is the right slope. Only peaks with higher slopes than threshold values will be kept.
    Ref: Gasc, A., Sueur, J., Pavoine, S., Pellens, R., & Grandcolas, P. (2013). Biodiversity sampling using a global acoustic approach: contrasting sites with microendemics in New Caledonia. PloS one, 8(5), e65311.
    """

    
    meanspec = np.array([np.mean(row) for row in spectro])

    if normalization:
         meanspec =  meanspec/np.max(meanspec)

    # Find peaks (with slopes)
    peaks_indices = np.r_[False, meanspec[1:] > np.array([x + slopes[0] for x in meanspec[:-1]])] & np.r_[meanspec[:-1] > np.array([y + slopes[1] for y in meanspec[1:]]), False]
    peaks_indices = peaks_indices.nonzero()[0].tolist()

    #peaks_indices = signal.argrelextrema(np.array(meanspec), np.greater)[0].tolist() # scipy method (without slope)


    # Remove peaks with difference of frequency < freqband
    nb_bin=next(i for i,v in enumerate(frequencies) if v > freqband) # number of consecutive index
    for consecutiveIndices in [np.arange(i, i+nb_bin) for i in peaks_indices]:
        if len(np.intersect1d(consecutiveIndices,peaks_indices))>1:
            # close values has been found
            maxi = np.intersect1d(consecutiveIndices,peaks_indices)[np.argmax([meanspec[f] for f in np.intersect1d(consecutiveIndices,peaks_indices)])]
            peaks_indices = [x for x in peaks_indices if x not in consecutiveIndices] # remove all inddices that are in consecutiveIndices
            peaks_indices.append(maxi) # append the max
    peaks_indices.sort()


    peak_freqs = [frequencies[p] for p in peaks_indices] # Frequencies of the peaks

    return len(peaks_indices)


def compute_ACI(spectro, wave, j_bin, sr):
    """
    Compute the Acoustic Complexity Index from the spectrogram of an audio signal.
    Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing activity of an avian community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11, 868-873.
    Ported from the soundecology R package.
    spectro: the spectrogram of the audio signal
    j_bin: temporal size of the frame (in samples)
    """
    
    times = range(0, spectro.shape[1], j_bin) # relevant time indices
    #times = range(0, spectro.shape[1]-10, j_bin) # alternative time indices to follow the R code

    jspecs = [np.array(spectro[:,i:i+j_bin]) for i in times]  # sub-spectros of temporal size j

    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs] 	# list of ACI values on each jspecs
    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values # return main (global) value, temporal values



    