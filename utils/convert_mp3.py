from genericpath import isfile
import os
import argparse
import pandas as pd
from pydub import AudioSegment

parser = argparse.ArgumentParser(description='Script to convert a whole folder of wav into MP3, slicing them as well into chunks')

parser.add_argument('-i', default=None, type=str, help='Path to folder to convert (will walk into it and find only the wav files)')
parser.add_argument('-p', default=None, type=str, help='Path to csv file corresponding to the input folder, with chunks')
parser.add_argument('-o', default=None, type=str, help='Path to output folder, will create a subfolder with the name of the top folder in the input (e.g. /bigdisk/folders/0001 will create a 0001 folder)')

args = parser.parse_args()

### Input folder 
path_audio_folder = args.i
_,topfolder = os.path.split(path_audio_folder)

### Create the output folders
outputbasefolder = args.o
outputfolder = os.path.join(outputbasefolder,topfolder)

os.makedirs(outputbasefolder,exist_ok=True)
os.makedirs(outputfolder,exist_ok=True)

### Open the pkl file 

meta_filename = args.p
print(f"Loading file {meta_filename}")
meta_file = pd.read_csv(meta_filename)
meta_file = meta_file.reset_index(drop=True)


for root, dirs, files in os.walk(path_audio_folder, topdown=False):
    for name in files:
        if name[-3:].casefold() == 'wav' and name[:2] != '._':
            curfile = (os.path.join(root,name))

            ### Searching this file in the pkl 
            Dffile = meta_file[meta_file['name']==name]
            
            nchunks = len(Dffile)
            if nchunks == 0:
                raise(f"File {curfile} not found !!!")
            ## How many chunks are they ? 
            #print(f"Found {nchunks} chunks for file {curfile}")

            ## Looping over chunks
            for curstart in Dffile['start']:
                curstart=int(curstart)

                ## reading  chunk
                wav_audio = AudioSegment.from_file(curfile, format="wav",start_second=curstart,duration=10)

                ## name of the mp3 file
                mp3_file = os.path.join(outputfolder,f"{os.path.splitext(name)[0]}_{curstart}.mp3")

                if not(os.path.isfile(mp3_file)):
                    #print(f"Converting {mp3_file}...")

                    file_handle = wav_audio.export(mp3_file,
                           format="mp3",
                           bitrate="128k")
                else:
                    print(f"File {mp3_file} already exists, skipping...")                                
                
            







print(outputfolder)