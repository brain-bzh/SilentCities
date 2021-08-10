import pandas as pd
from tqdm import tqdm
from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description='Silent City Meta data genetor')

parser.add_argument('--path_pkl', default='/Users/nicolas/Documents/SilentCities/SilentCities/meta_silentcities/site/0001.pkl', type=str, help='Path to pkl metafile of one site')
parser.add_argument('--save_path', default='/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/0001_new.pkl', type=str, help='Path to new pkl metafile')
parser.add_argument('--start_date', default='20200501_100100', type=str, help='start date in format YYYYMMDD_HHMMSS')
parser.add_argument('--stop_date', default='20200502_100100', type=str, help='stop date in format YYYYMMDD_HHMMSS')
args = parser.parse_args()

Df = pd.read_pickle(args.path_pkl)
Df = Df.sort_values('datetime').reset_index(drop=True)
start = datetime.strptime(args.start_date, '%Y%m%d_%H%M%S')
stop = datetime.strptime(args.stop_date, '%Y%m%d_%H%M%S')
print(Df)

def pop_dataframe(df, values, axis=0):
    if axis == 0:
        if isinstance(values, (list, tuple)):
            popped_rows = df.loc[values]
            df.drop(values, axis=0, inplace=True)
            return popped_rows
        elif isinstance(values, (int)):
            popped_row = df.loc[values].to_frame().T
            df.drop(values, axis=0, inplace=True)
            return popped_row
        else:
            print('values parameter needs to be a list, tuple or int.')
    elif axis == 1:
        # current df.pop(values) logic here
        return df.pop(values)

print(start < stop)
pop = []
for idx, date in zip(tqdm(Df.index), Df['datetime']):
    if date > start and date < stop:
        pass
    else:
        pop.append(idx)
print(pop)
pop_dataframe(Df, pop)
Df = Df.sort_values('datetime').reset_index(drop=True)
Df.to_pickle(args.save_path)
print(Df)







#
# Df = pd.read_excel('/Users/nicolas/Documents/SilentCities/SilentCities/meta_silentcities/Table_Sound_Validation_sortedJF_20210408.xlsx')
# print(Df)
#
# list_site = []
# dict_file = {}
# current_site = 0
# for idx, (filename, site) in enumerate(zip(tqdm(Df['Name_recording']), Df['Site'])):
#     print(idx, filename, site)
#     if current_site != site:
#         list_site.append(site)
#         dict_file[str('{:04d}'.format(list_site[-1]))] = [filename]
#         current_site = site
#     else : dict_file[str('{:04d}'.format(list_site[-1]))].append(filename)
#
# print(list_site)
# print(dict_file)
#

