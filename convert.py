import os
import torch
import numpy as np
import multiprocessing
import pandas as pd
import argparse
from tqdm import tqdm
from audioset_tagging_cnn.inference import audio_tagging,prepare_model
import utils.utils
from analysis.tagging_validation import tagging_validate
from utils.parameters import len_audio_s
import datetime
import librosa
import tempfile
from torch.utils.data import Dataset
from pydub import AudioSegment
import soundfile as sf
import shutil
import tarfile
import glob
import datetime

#start of timer 
start_time = datetime.datetime.now()


#python convert.py --metadata_folder /bigdisk1/meta_silentcities/dbfiles --site 0065 --database /bigdisk1/database_pross.pkl --batch_size 16 --toflac /bigdisk1/flac
defult_tmp_dir = tempfile._get_default_tempdir()
#### a comment 
NUM_CORE = multiprocessing.cpu_count()
parser = argparse.ArgumentParser(
    description='Converting files to FLAC')
parser.add_argument('--length', default=10, type=int, help='Segment length')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--metadata_folder', default=None,
                    type=str, help='folder with all metadata')
parser.add_argument('--results_folder', default=None,
                    type=str, help='folder with acoustic measurements files with speech')
parser.add_argument('--site', default=None, type=str, help='site to process')
parser.add_argument('--toflac', default=None, type=str,
                    help='Path to folder for FLAC conversion, (Default : no conversion). Will create a subfolder with site name')
parser.add_argument('--database', default=None, type=str,
                    help='Path to metadata (given by metadata_site.csv)')

parser.add_argument('--nocuda', action='store_false',
                    help='Do not use the GPU for acceleration')
parser.add_argument('--preload', action='store_true',
                    help='preload wave files into RAM')
parser.add_argument('--ncpu',default=NUM_CORE-1,type=int,
                    help='Number of CPUs for parallelization')
parser.add_argument('--destdir', default='/nasbrain/datasets/silentcities', type=str,
                    help='Path to destination folder')
args = parser.parse_args()

## find the path to the data using the site name 
if os.path.isdir(os.path.join('/bigdisk1/silentcities',args.site)):
    sitefolder = os.path.join('/bigdisk1/silentcities',args.site)
    print(f"Data for site {args.site} found in /bigdisk1 ")
elif os.path.isdir(os.path.join('/bigdisk2/silentcities',args.site)):
    sitefolder = os.path.join('/bigdisk2/silentcities',args.site)
    print(f"Data for site {args.site} found in /bigdisk2 ")
elif os.path.isdir(os.path.join('/users/local/bigdisk1/silentcities',args.site)):
    sitefolder = os.path.join('/users/local/bigdisk1/silentcities',args.site)
    print(f"Data for site {args.site} found in /users/local/bigdisk1 ")
elif os.path.isdir(os.path.join('/users/local/bigdisk2/silentcities',args.site)):
    sitefolder = os.path.join('/users/local/bigdisk2/silentcities',args.site)
    print(f"Data for site {args.site} found in /users/local/bigdisk2 ")
else:
    raise ValueError(f"Data for site {args.site} not found either in /bigdisk1 or /bigdisk2")


## Use the relative path to /bigdisk if it's not mounted

if not(os.path.isdir('/bigdisk1')):
    args.metadata_folder = args.metadata_folder.replace('/bigdisk1','/users/local/bigdisk1')
    args.toflac = args.metadata_folder.replace('/bigdisk1','/users/local/bigdisk1')
    args.database = args.database.replace('/bigdisk1','/users/local/bigdisk1')
    args.results_folder = args.results_folder.replace('/bigdisk1','/users/local/bigdisk1')
    print("replaced the bigdisk1 path with /users/local/bigdisk1")


## utility functions


class Convert_Dataset(Dataset):
    def __init__(self, meta_dataloader,toflac,preload=False):
        self.meta = meta_dataloader
        self.toflac = toflac ### this must be the output site folder
        print(f"Temp folder for wav files for mp3 conversion : {defult_tmp_dir}")
        self.preload = preload
        self.data = []
        if preload:
            print('Preloading dataset...')
            for idx,curfile in tqdm(enumerate(self.meta['filename'])):
                filename = curfile

                wav_o, sr = librosa.load(filename, sr=None, mono=True,offset=self.meta['start'][idx], duration=len_audio_s)
                self.data.append((wav_o,sr))


    def __getitem__(self, idx):
        filename = self.meta['filename'][idx]

        if self.preload:
            wav_o,sr = self.data[idx]
        else:
            wav_o, sr = librosa.load(filename, sr=None, mono=True,
                              offset=self.meta['start'][idx], duration=len_audio_s)        
        curdate = self.meta['date'][idx].strftime('%Y%m%d_%H%M%S')
        ## name of the flac file
        flacfile = os.path.join(self.toflac,self.meta['flacfile'][idx])
        #print(flacfile)
        if not(os.path.isfile(flacfile)):
            #print(f"Converting {flacfile}...")                
            ## Converting current chunk to temporary wav file in a temp folder

            temp_name = os.path.join(defult_tmp_dir,next(tempfile._get_candidate_names()) + '.wav')
            sf.write(temp_name,wav_o,sr)

            ## reading  chunk
            wav_audio = AudioSegment.from_file(temp_name, format="wav")

            file_handle = wav_audio.export(flacfile,
                    format="flac")

            os.remove(temp_name)
        
        
        return {'name': os.path.basename(filename), 'start': self.meta['start'][idx],
                                                        'date': curdate}

    def __len__(self):
        return len(self.meta['filename'])
    
def get_dataloader_site_fromresults(site_ID, path_wavfile, results_path, meta_path,meta_site, database, batch_size=6,flacfolder = None,ncpu=NUM_CORE,preload=False):
    partIDidx = database[database.partID == int(site_ID)].index[0]

    filelist = []
    for root, dirs, files in os.walk(path_wavfile, topdown=False):
        for name in files:
            if name[-3:].casefold() == 'wav' and name[:2] != '._':
                filelist.append(os.path.join(root, name))
    
    filelist_base = [os.path.basename(file_) for file_ in filelist]

    resultsfile = pd.read_csv(os.path.join(results_path,f"partID{args.site[1:]}.csv"))
    speechfiles = resultsfile.loc[resultsfile['reject_speech'] == 1,'name'].to_list()

    print(f"Found {len(np.unique(speechfiles))} speech files in results file")

    #remove files in filelist that are in speechfiles
    filelist_base = [file_ for file_ in filelist_base if file_ not in speechfiles]

    meta_dataloader = pd.DataFrame(
        columns=['filename', 'sr', 'start', 'stop'])
    
    for idx, wavfile in enumerate(tqdm(meta_site['filename'])):

        len_file = meta_site['length'][idx]
        sr_in = meta_site['sr'][idx]
        if wavfile in speechfiles:
            print(f"Speech file {wavfile} rejected")
            continue

        duration = len_file/sr_in

        nb_win = int(duration // len_audio_s )

        for win in range(nb_win):
            delta = datetime.timedelta(seconds=int((win*len_audio_s)))
            datevalue = meta_site['datetime'][idx] + delta
            datestring = datevalue.strftime('%Y%m%d_%H%M%S')
            filename_flac = "partID" + args.site[1:]+'-'+datestring+'.flac'
            resultsfile.loc[resultsfile['datetime'] == datestring, 'filename_short'] = filename_flac
            curmeta = pd.DataFrame.from_dict({'filename': [filelist[filelist_base.index(wavfile)]], 'sr': [sr_in], 'start': (
                [win*len_audio_s]), 'stop': [((win+1)*len_audio_s)], 'len': [len_file], 'date': datevalue, 'flacfile':filename_flac})
            meta_dataloader = pd.concat([meta_dataloader,curmeta], ignore_index=True)                
        
    meta_dataloader.to_pickle(os.path.join(meta_path, site_ID+'_metaloader_speechrejected.pkl'))
    print(meta_dataloader)
    # rename the "name" column into "original_filename"
    resultsfile = resultsfile.rename(columns={'name':'original_filename'})
    #rename the filename_short column of resultsfiles into name
    resultsfile = resultsfile.rename(columns={'filename_short':'name'})
    #change the order of columns put the name column first 
    cols = resultsfile.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    resultsfile = resultsfile[cols]
    # drop the lines corresponding to reject_speech == 1
    resultsfile = resultsfile.loc[resultsfile['reject_speech'] == 0]
    # drop the reject_speech column
    resultsfile = resultsfile.drop(columns=['reject_speech','start'])
    # save the resultsfile, compressed
    print(f"Saving results file to {os.path.join(args.toflac,args.site[1:],f'partID{args.site[1:]}.csv.gz')}")
    resultsfile.to_csv(os.path.join(args.toflac,args.site[1:],f"partID{args.site[1:]}.csv.gz"),index=False,compression='gzip')
    

    site_set = Convert_Dataset(meta_dataloader=meta_dataloader.reset_index(drop=True),
    toflac=flacfolder,preload=preload)
    site_set = torch.utils.data.DataLoader(
        site_set, batch_size=batch_size, shuffle=False, num_workers=ncpu)

    return site_set

if args.metadata_folder is None:
    raise(AttributeError("Must provide metadata folder"))
if args.site is None:
    raise(AttributeError("Must provide site"))

flacfolder = args.toflac
if not(flacfolder is None):
    print("Will convert to FLAC...")
    flacfolder = os.path.join(flacfolder,args.site[1:],'flac')
    print(f"Creating folder {flacfolder}")
    os.makedirs(flacfolder,exist_ok=True)

if args.database is None:
    raise(AttributeError("Must provide database"))

try:
    DATABASE = pd.read_pickle(args.database).reset_index(drop=True)
    if not(DATABASE.partID.dtype == 'int64'):
        raise('error part ID (must be int64) not {}'.format(DATABASE.partID.dtype))
except:
    DATABASE = pd.read_csv(args.database).reset_index(drop=True)
meta_site = pd.read_pickle(os.path.join(
    args.metadata_folder, args.site+'.pkl')).reset_index(drop=True)

site_set = get_dataloader_site_fromresults(
    site_ID=args.site, path_wavfile=sitefolder, results_path=args.results_folder,meta_site=meta_site,meta_path=args.metadata_folder,database = DATABASE,batch_size=args.batch_size,flacfolder=flacfolder,ncpu=args.ncpu,preload=args.preload)

print("FLAC conversion...")


for batch_idx, info in enumerate(tqdm(site_set)):

    pass

print("Flac conversion Done !")


# create gz archive with files from the flacfolder with flac files, so that each archive is not larger than 4GB
# create a folder for the archives
filesperarchive = 10000
archive_folder = os.path.join(args.toflac,args.site[1:],'audio')
print(f"Creating folder {archive_folder}")
os.makedirs(archive_folder,exist_ok=True)
# get the list of flac files
flac_files = glob.glob(os.path.join(flacfolder,'*.flac'))
# get the number of files
nb_files = len(flac_files)
# get the number of archives
nb_archives = int(nb_files/filesperarchive) + 1

print(f"There is a total of {nb_files} FLAC files to compress. I will do {nb_archives} archives, each with {filesperarchive} files")
# create the archives
ndigits = len(str(nb_archives))
for idx in tqdm(range(nb_archives)):
    # get the list of files to be archived
    files_to_archive = flac_files[idx*filesperarchive:(idx+1)*filesperarchive]
    # create the name of the archive
    archive_name = os.path.join(archive_folder,f"partID{args.site[1:]}_{str(idx+1).zfill(ndigits)}.tar.gz")
    # create the archive
    with tarfile.open(archive_name, "w:gz") as tar:
        for file_ in files_to_archive:
            tar.add(file_, arcname=os.path.basename(file_))
    # remove the files
    for file_ in files_to_archive:
        os.remove(file_)

print('Removing flac folder')
# remove the flac folder
os.rmdir(flacfolder)

print('Copying all the results to the nas')
destdir = args.destdir
sitefolder = os.path.join(args.toflac,args.site[1:])
os.makedirs(destdir,exist_ok=True)

# copy the whole sitefolder to destdir
shutil.copytree(sitefolder,destdir)

# time elapsed
elapsed = time.time() - start_time
print(f"Time elapsed: {elapsed:.2f} s for {nb_files} files")

print('Finished !')