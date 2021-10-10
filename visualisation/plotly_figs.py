import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import librosa
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0, "../")
from utils import librosa_display as ld


def get_map_fig(database):
    fig = px.scatter_mapbox(database, lat="lat", lon="lng", color = "s_statsmean",hover_data=['partID', 'city', 'country', "recorder"], zoom=1, height = 600, size_max=10)
    fig.update_layout(
        mapbox_style="open-street-map",autosize=True,showlegend=False,
                margin=go.layout.Margin(l=0, r=0, t=0, b=0))
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_colorbar=dict(
        title="<b>Indicateur</b>",titleside='right',
                        thicknessmode="pixels", thickness=10,
                        lenmode="pixels", len=250,
                        yanchor="bottom", y=0.1,
                        xanchor="right", x=1.0
                        ))
    return fig

def get_heatmaps(site, path):
    global data

    macro_cat = {'geophony':['Wind', 'Rain', 'River', 'Wave', 'Thunder'],
                 'biophony': ['Bird', 'Amphibian', 'Insect', 'Mammal', 'Reptile'], 
                 'anthropophony': ['Walking', 'Cycling', 'Beep', 'Car', 'Car honk', 'Motorbike', 'Plane', 'Helicoptere', 'Boat', 'Others_motors', 'Shoot', 'Bell', 'Talking', 'Music', 'Dog bark', 'Kitchen sounds', 'Rolling shutter']}
    try:
        data = pd.read_csv(os.path.join(path, f'tagging_site_{site:04d}.csv'))
    except:
        data = pd.read_csv(os.path.join(path, f'results_{site:04d}.csv'))


    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("<b>Anthropophonie</b>", "<b>Géophonie</b>", "<b>Biophonie</b>", ""), vertical_spacing=0.04)

    fig.add_trace(go.Heatmap(x = data['datetime'], y = macro_cat['geophony'], z = data[[f'tag_{idx}' for idx in macro_cat['geophony']]].T, coloraxis='coloraxis', name='Géophonie',colorscale='Hot'), row=2, col=1)
    fig.add_trace(go.Heatmap(x = data['datetime'], y = macro_cat['biophony'], z = data[[f'tag_{idx}' for idx in macro_cat['biophony']]].T, coloraxis='coloraxis', name='Biophonie',colorscale='Hot'), row=3, col=1)
    fig.add_trace(go.Heatmap(x = data['datetime'], y = macro_cat['anthropophony'], z = data[[f'tag_{idx}' for idx in macro_cat['anthropophony']]].T, coloraxis='coloraxis', name='Anthropophonie', colorscale='viridis'), row=1, col=1)
    # for idx in range(3):
    #     fig.add_trace(go.Scattergl(x = data['datetime'], y = data['tag_{}'.format(macro_cat['geophony'][idx])], name=macro_cat['geophony'][idx]), row=1, col=1)
    #     fig.add_trace(go.Scattergl(x = data['datetime'], y = data['tag_{}'.format(macro_cat['biophony'][idx])],  name=macro_cat['biophony'][idx]), row=2, col=1)
    #     fig.add_trace(go.Scattergl(x = data['datetime'], y = data['tag_{}'.format(macro_cat['anthropophony'][idx])],  name=macro_cat['anthropophony'][idx]), row=3, col=1)
    fig.add_trace(go.Scattergl(x = data['datetime'], y = data['anthropophony'],name='Anthropophonie', line = dict(color='black'), opacity=0.5), row=4, col=1)
    fig.add_trace(go.Scattergl(x = data['datetime'], y = data['geophony'],name="Géophonie", line = dict(color='blue'), opacity=0.5), row=4, col=1)
    fig.add_trace(go.Scattergl(x = data['datetime'], y = data['biophony'], name= 'Biophonie', line = dict(color='green'), opacity=0.5), row=4, col=1)
    
   
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0},coloraxis_colorbar=dict(
        title="<b>Probabilité</b>",titleside='right',
                        thicknessmode="pixels", thickness=30,
                        lenmode="pixels", len=400,
                        yanchor="bottom", y=0.0,
                        xanchor="right", x=1.1
                        ),yaxis = {'fixedrange': True}, height=800)
    fig.update_yaxes({'fixedrange': True}, row=2, col=1)
    fig.update_yaxes({'fixedrange': True}, row=3, col=1)
    fig.update_yaxes({'fixedrange': True}, row=4, col=1)

    return fig, data

def get_sample_fig(site, file, path):
    path_audio = os.path.join(path, f'{site:04d}', file)
    audio, sr = librosa.load(path_audio, sr = 44100)

    audio = audio[:int(10*sr)]
    N = len(audio)
    Zxx = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=20000)
    t, f, Zxx = ld.specshow(Zxx, sr = sr, fmax = 20000,y_axis='mel', x_axis='time', ref = np.max)

    # f, t, Zxx = stft(audio, sr, nperseg=1024, noverlap=512, nfft=2048)
    # cmap = plt.get_cmap('jet')
    # rgba_img = cmap(10*np.log10(Zxx+1e-12))
    # rgb_img = np.delete(rgba_img, 3, 2)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("<b>Représentation temporelle</b>", "<b>Spectrogramme</b>"), vertical_spacing=0.04)
    fig.add_trace(go.Scattergl(x = (np.arange(N)/sr)[::100], y = audio[::100], name="Représentation temporelle"), row=1, col=1)
    fig.add_trace(go.Heatmap(x = t, y = f, z = 10*np.log10(Zxx+1e-12), colorscale='thermal',zauto=False, zmin = -60, zmax=0, name='Spectrogramme'), row=2, col=1)
    fig.update_yaxes(title = 'Fréquence (Hz)',type='log', range=[2,4],row=2, col=1)
    fig.update_yaxes(title = 'Amplitude (-)',row=1, col=1)
    fig.update_xaxes(title = 'Temps (s)',row=2, col=1)
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    return fig, path_audio
    



    






if __name__ == '__main__':
    # fig = get_heatmaps(25)
    # fig.show()
    fig = get_sample_fig(61, "5E81A700.WAV")
    fig.show()
