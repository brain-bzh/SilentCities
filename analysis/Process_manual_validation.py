import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from py import process
import os 
bigdiskpath = '/home/nfarrugi/bigdisk'

bigdisk1path = bigdiskpath + '1'
bigdisk2path = bigdiskpath + '2'

xlspath = '/home/nfarrugi/SilentCities/papers/SciDataPaper/analysis/identification_manuelle/Manual_validation_indic_corrected_blank.xlsx'
xlspath_out = '/home/nfarrugi/SilentCities/papers/SciDataPaper/analysis/identification_manuelle/Manual_validation_indic_corrected_final_max.xlsx'

Df = pd.read_excel(xlspath,index_col=0)

## Detect lines with missing values and extract the corresponding Dataframe
Df_missing = Df.loc[pd.isna(Df['tag_Bell'])]

missingsites = Df_missing['Site'].unique()

### match all the results output and fill the missing columns

for site in missingsites:
    cursite = "%04d" % (site)
    print(f"Site {cursite}")

    allfiles = Df_missing.loc[Df_missing['Site']==site,'Name_recording']
    
    # Load the processed file for this site
    processedDf = pd.read_csv(os.path.join(bigdisk2path,'meta_silentcities','site',f"results_{cursite}.csv"))
    processedDf['name2'] = [i.casefold() for i in processedDf['name']]
    for curfile in allfiles:
        if curfile.casefold()[-4:] != '.wav':
            curfile_ext = curfile.casefold()+'.wav'
        else:
            curfile_ext = curfile
        subDf = processedDf.loc[processedDf['name2']==curfile_ext]
        if len(subDf)==0:
            subDf = processedDf.loc[processedDf['name']==curfile_ext]
        
        if len(subDf)>0:
            lst_tag = subDf.columns[3:-1]
            #print(f"Site {cursite} Found file {curfile} setting values..")
            for lst_tag_idx in lst_tag:
                if (lst_tag_idx[:3]=='tag') or (lst_tag_idx=='biophony') or (lst_tag_idx=='geophony') or (lst_tag_idx=='anthropophony'):
                    Df_missing.loc[((Df_missing['Site']==site) & (Df_missing['Name_recording']==curfile)),lst_tag_idx] = np.max(subDf[lst_tag_idx].to_numpy())
                else:
                    Df_missing.loc[((Df_missing['Site']==site) & (Df_missing['Name_recording']==curfile)),lst_tag_idx] = np.mean(subDf[lst_tag_idx].to_numpy())

                        
        
Df_part = Df.dropna(axis=0,how='any')
Df_final = pd.concat([Df_part,Df_missing]).sort_values(by=['Site'])
Df_final.to_excel(xlspath_out)
        

















#Df_missing[['Site','Name_recording']].to_csv('missingfiles.csv',index=False)

#print(Df_missing['Site'].unique())

