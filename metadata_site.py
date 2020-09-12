import os
import pandas as pd
import datetime
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Silent City Meta data genetor')

parser.add_argument('--folder', default=None, type=str, help='Path to folder with meta data')
parser.add_argument('--save_path', default=None, type=str, help='Path to save meta data by site')
parser.add_argument('--database', default=None, help='database (csv)')
parser.add_argument('--verbose', action='store_true', help='Verbose (default False = nothing printed)')


args = parser.parse_args()
if args.folder is None:    
    raise(AttributeError("Must provide either a file or a folder"))
if not(args.database[-3:] == 'csv') and args.database is not None:
    raise(AttributeError("Must provide CSV database"))
DATABASE = pd.read_csv(args.database)
DATABASE = DATABASE.loc[:, ~DATABASE.columns.str.contains('^Unnamed')]

DATABASE['mar'] = None
DATABASE['apr'] = None
DATABASE['may'] = None
DATABASE['jun'] = None
DATABASE['jul'] = None
DATABASE['aug'] = None
DATABASE['error'] = None
DATABASE['nb_file'] = None
DATABASE['sr'] = None
DATABASE['len'] = None
DATABASE['min_dB'] = None
DATABASE['ref_file'] = None
DATABASE['error_date'] = False

nb_rec_month = {'03' : 4464,
                '04' : 4320,
                '05' : 4464,
                '06' : 4320,
                '07' : 4464,
                '08' : 4464}
month_cov = {'03' : 'mar',
                '04' : 'apr',
                '05' : 'may',
                '06' : 'jun',
                '07' : 'jul',
                '08' : 'aug'}

print(DATABASE.columns)

if not(os.path.exists(args.save_path)):
    os.mkdir(args.save_path)

filelist = []

for root, dirs, files in os.walk(args.folder, topdown=False):
    for name in files:
        if not('error' in name) and name[-3:].casefold() == 'pkl':
            filelist.append(os.path.join(root, name))
print(filelist)            
filelist.sort()

print(filelist)


def process_site(database, file_):
    partID = int(os.path.basename(file_[:-4]))
    error_date = False
    df = pd.read_pickle(file_)
    df_error = pd.read_pickle(file_[:-4]+'_error.pkl')
    nb_error = len(df_error)
    nb_file = len(df)
    if nb_file > 1 :
        nb_month = {'03' : 0,
                    '04' : 0,
                    '05' : 0,
                    '06' : 0,
                    '07' : 0,
                    '08' : 0}


        database['error'][partID-1] = nb_error
        database['nb_file'][partID-1] = nb_file
        database['sr'][partID-1] = df['sr'].mean()
        database['len'][partID-1] = df['length'].mean() / database['sr'][partID-1]

        for idx, filename in enumerate(df['filename']):
            if idx == 0:
                ref_dB = df['dB'][idx]
                ref_dB_file = filename
                

            date = df['datetime'][idx]
            try:
                nb_month['%02d' % date.month] += 1
            except:
                error_date = True
            if ref_dB > df['dB'][idx]:
                ref_dB_file = filename
                ref_dB = df['dB'][idx]

        database['min_dB'][partID-1] = ref_dB
        database['ref_file'][partID-1] = ref_dB_file
        if error_date:
            database['error_date'][partID-1] = True
        for key in nb_month.keys():
            database[month_cov[key]][partID-1] =  nb_month[key]/nb_rec_month[key]*100

    print(file_)
    
    return database
        

for file_ in filelist:
    DATABASE = process_site(DATABASE.copy(deep = True), file_)

DATABASE.to_pickle('database_pross.pkl')
DATABASE.to_csv('database_pross.csv')