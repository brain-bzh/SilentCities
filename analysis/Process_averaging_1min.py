import pandas as pd
import numpy as np


list_site = [4, 11, 25, 36, 38, 48, 52, 61, 62, 77, 87, 115, 120, 121, 132, 153, 158, 159, 190, 229, 234, 276, 292, 346, 371, 388]


for site in list_site:

    Tagg = pd.read_csv(f'/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/NEW2/tagging_site_{site:04d}.csv')
    # Tagg = Tagg.sort_values('name').reset_index(drop=True)
    indic = pd.read_csv(f'/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/NEW2/site_{site:04d}.csv')
    # indic = 

    file_save_name = f'/Users/nicolas/Documents/SilentCities/SilentCities/ecoacoustique/NEW2/tagging_site_av_{site:04d}.csv'

    col = list(Tagg.columns)
    col[1] = 'datetime'
    for idx in range(2, len(col)-3):
        col[idx] = col[idx][4:]
    seen = {}

    filemane = [seen.setdefault(x, x) for x in list(Tagg['name']) if x not in seen] 

    out = pd.DataFrame(data = {'name': filemane}, columns=col)

    curr_name = ""
    idx = 0
    for filemane in Tagg['name']:
        if curr_name != filemane:
            list_site.append(filemane)
            idx_val = list(Tagg['name'] == str(filemane))

            for lst_tag_idx in col :
                if lst_tag_idx == 'name':
                    out[lst_tag_idx][idx] = filemane
                elif lst_tag_idx == 'datetime':
                    out[lst_tag_idx][idx] = indic[lst_tag_idx][idx_val].iloc[0]
                
                elif lst_tag_idx == 'geophony' or lst_tag_idx == 'biophony' or lst_tag_idx == 'anthropophony' :
                    out[lst_tag_idx][idx] = np.mean(Tagg[f'{lst_tag_idx}'][idx_val])

                else:
                    out[lst_tag_idx][idx] = np.mean(Tagg[f'tag_{lst_tag_idx}'][idx_val])

            idx += 1
            
        else:
            pass

        curr_name = filemane
    out.to_csv(file_save_name)
    print(out)



