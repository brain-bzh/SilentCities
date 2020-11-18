import os

path_audio_folder = '/bigdisk1/silentcities/'

for root, dirs, files in os.walk(path_audio_folder, topdown=False):
    for name in files:
        if name[-3:].casefold() == 'wav' and name[:2] != '._':
            print(os.path.join(root,name))