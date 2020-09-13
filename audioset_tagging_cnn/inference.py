### modified from https://github.com/qiuqiangkong/audioset_tagging_cnn
import os
import sys
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
from .utilities import create_folder, get_filename
from .models import *

from .pytorch_utils import move_data_to_device
from .config import sample_rate,classes_num,labels
from scipy.signal import resample 

def audio_tagging(waveform, checkpoint_path,offset=None,duration=None,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000,model_type="ResNet22",usecuda=False):
    """Inference audio tagging result of an audio clip.
    """

    device = torch.device('cuda') if usecuda and torch.cuda.is_available() else torch.device('cpu')

    global sample_rate
    global classes_num
    global labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    #print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
        waveform.to(device)

    
    # Forward
    # print(waveform.size())
    model.eval()
    batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    #for k in range(10):
    #    print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
    #        clipwise_output[sorted_indexes[k]]))

    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()
        #print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels, sorted_indexes, embedding