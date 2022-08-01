import sys
sys.path.insert(0, "../")

import os 
from glob import glob
# import argparse
import csv

import numpy as np
import pandas as pd
import visualisation.plotly_figs as pf
# from pydub import AudioSegment
from datetime import datetime


PATH_MP3 = '/Volumes/LaCie/silent_cities/mp3_sl/'
PATH_DATABASE = "/Volumes/LaCie/silent_cities/public_final_metadata_geo_stats.csv"
PATH_TAGSITE = "/Volumes/LaCie/silent_cities/site/"

# PATH_MP3 = '/Users/nicolas/Downloads/mp3_sl/'
# PATH_DATABASE = "/Users/nicolas/Documents/SilentCities/database/public_final_metadata_geo_stats.csv"
# PATH_TAGSITE = "/Users/nicolas/Documents/SilentCities/database/meta_silentcities"

#### Initialization
database = pd.read_csv(PATH_DATABASE)

# get available site
available_site_mp3 = glob(os.path.join(PATH_MP3, '*/'))
available_site_mp3 = np.sort([int(i[-5:-1]) for i in available_site_mp3])
available_site_process = glob(os.path.join(PATH_TAGSITE, 'results*'))
available_site_process = np.sort([int(i[-8:-4]) for i in available_site_process])
available_site = list(set(available_site_process) & set(available_site_mp3))
database = database[database['partID'].isin(available_site)].reset_index(drop=True)

print(available_site_mp3)

figindic, data = pf.get_heatmaps_nd(1, path=PATH_TAGSITE, title = 'France')
figindic.write_html('001.html')
figindic.show()


