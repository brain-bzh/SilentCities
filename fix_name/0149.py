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
    curdate_time = datetime.datetime.strptime('{}_{}'.format(name[9:17],name[18:24]),"%Y%m%d_%H%M%S")
    if len(name) == 30:
        delta = datetime.timedelta(minutes=int(name[-5])-1)
    elif len(name) == 31:
        delta = datetime.timedelta(minutes=9)
    
    ttime = curdate_time + delta

    strtime = ttime.strftime("%Y%m%d_%H%M%S")


    

    if name[-6] == "-" or name[-7] == "-" :
        os.rename(path_name, os.path.join(path, name[:9]+ strtime + '.wav'))
    else:
        pass

