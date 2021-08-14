import os
import numpy as np
import pandas as pd


manual_validation = pd.read_excel('/Users/nicolas/Documents/SilentCities/SilentCities/meta_silentcities/Table_Sound_Validation_sortedJF_20210408.xlsx')
manual_validation = manual_validation.sort_values('Site').reset_index(drop=True)

list_site = [4, 11, 25, 36, 38, 48, 52, 61, 62, 77, 87, 115, 120, 121, 132, 153, 158, 159, 190, 229, 234, 276, 292, 346, 371, 388]


indic = {'dB': [], 'ndsi_N': [], 'aci_N': [],
                        'BI_N': [], 'EAS_N': [],
                        'ECV_N': [], 'EPS_N': [] ,'ndsi_W': [], 'aci_W': [],
                        'BI_W': [], 'EAS_W': [],
                        'ECV_W': [], 'EPS_W': [], 'ACT':[],
                    'POWERB_126':[], 'POWERB_251':[], 'POWERB_501':[], 'POWERB_1k':[], 'POWERB_2k':[], 'POWERB_4k':[], 'POWERB_8k':[], 'POWERB_16k':[]}



lst_indic = indic.keys()


for cur_idx in lst_indic:
    manual_validation[cur_idx] = None

add_col = True
for site in list_site:

    Df = pd.read_csv(f'/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/NEW3/site_{site:04d}.csv')
    Tagg = pd.read_csv(f'/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/NEW2/tagging_site_{site:04d}.csv')
    if add_col:
        lst_tag = Tagg.columns[2:]
        for lst_tag_idx in lst_tag : manual_validation[lst_tag_idx] = None 
        add_col = False

    Df['name'] = [oo.casefold() for oo in Df['name'] ]

    
    for idx, name in enumerate(manual_validation['Name_recording']):

        try:
            if name.casefold()[-4:] != '.wav':
                name = name+'.wav'
            idx_val = list(Df['name'] == str(name).casefold() )

            if idx_val.count(True) == 0:
                pass
            else:
                for cur_idx in lst_indic:
                    manual_validation[cur_idx][idx] = np.mean(Df[cur_idx][idx_val])
                
                
                
                for lst_tag_idx in lst_tag :
                    manual_validation[lst_tag_idx][idx] = np.mean(Tagg[lst_tag_idx][idx_val])

        except ValueError:

            pass

print(manual_validation)        
manual_validation.to_excel('Manual_validation_indic.xlsx')
manual_validation.to_csv('Manual_validation_indic.csv')


# for site in list_site:

#     Tagg = pd.read_csv(f'/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/NEW2/tagging_site_{site:04d}.csv')
#     Tagg = Tagg.sort_values('name').reset_index(drop=True)
    
#     list_name = []
#     curr_name = 0
#     for idx, filemane in enumerate(Tagg['name']):
#         if curr_name != filemane:
#             list_site.append(filemane)

#         curr_name = filemane

#     file_save_name = f'/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/NEW2/tagging_site_av_{site:04d}.csv'

#     col = Tagg.columns
#     out = pd.DataFrame(columns=col)

#     for idx, file in enumerate(list_site):
#         for keys in col:




