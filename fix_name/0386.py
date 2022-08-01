import os
import argparse
import datetime


parser = argparse.ArgumentParser(description='Silent City Meta data genetor')
parser.add_argument('--folder', default=None, type=str, help='Path to folder with wavefiles, will walk through subfolders')

args = parser.parse_args()
folder =  args.folder


filelist = []

for root, dirs, files in os.walk(folder, topdown=False):
    for name in files:
        if name[-3:].casefold() == 'wav' and name[:2] != '._':
            filelist.append(os.path.join(root, name))



for path_name in filelist:
    name = os.path.basename(path_name)
    path = os.path.dirname(path_name)
    print(name[5:8+6+1+5])
   
    if name[0] == "B":
        os.rename(path_name, os.path.join(path, name[5:8+6+1+5] + '.wav'))
    else:
        pass

