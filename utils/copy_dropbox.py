from zipfile import ZipFile
import pandas as pd 
import os
from datetime import datetime
now = datetime.now()
date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
# %% [markdown]
# 1. For each folder or dropbox, search for an existing folder on bigdisk1 or bigdisk2. 
# 2. If the folder is not found, this is a new site. 

bigdisk1 = '/bigdisk1'
bigdisk2 = '/bigdisk2'

dropboxfolder = '/bigdisk2/new/dropbox/SilentCitiesData/'

newsites = []
bigdisk1sites = []
bigdisk2sites = []

for curdir in os.listdir(dropboxfolder):
    if os.path.isdir(os.path.join(bigdisk1,'silentcities',curdir)):
        nfiles = len(os.listdir(os.path.join(bigdisk1,'silentcities',curdir)))
        print(f"{curdir} found on bigdisk1 with {nfiles} files or directories inside")
        bigdisk1sites.append(curdir)    
    elif os.path.isdir(os.path.join(bigdisk2,'silentcities',curdir)):
        nfiles = len(os.listdir(os.path.join(bigdisk2,'silentcities',curdir)))
        print(f"{curdir} found on bigdisk2 with {nfiles} files or directories inside")
        bigdisk2sites.append(curdir)
    else:
        newsites.append(curdir)
        print(f"{curdir} is a new site")

# %% [markdown]
# 3. Parse the list of new sites and put them on bigdisk2, extract them flat. Save the list of new sites as sites for which we will need to restart "metadata_file"
# 

# %%
def extract_list_from_zip(zipfile,destdir,filelist=None):
    ### filelist is a subset (potentially all) from the list of files in zipfile
    ### If not, take the whole list of files from the zip

    os.makedirs(destdir,exist_ok=True)

    z = ZipFile(zipfile)

    if filelist is None:
        filelist = z.namelist()
        print("Extracting all files...")
    
    print(f"Extracting {len(filelist)} files")

    for curfile in filelist:
        extension = os.path.splitext(curfile)[1]
        
        if not(extension.casefold() == '.wav'):
            print("not a wav file")
            continue
        # In case in the filelist there are paths
        filenameonly = os.path.split(curfile)[1]        

        # extract a specific file from the zip container
        f = z.open(curfile)

        # save the extracted file 
        content = f.read()
        f = open(os.path.join(destdir,filenameonly), 'wb')
        f.write(content)
        f.close()

Df=  pd.DataFrame(newsites)

Df.to_csv(f"/bigdisk2/new/dropbox/newsites_{date_time}.csv",index=False,header=False)

Df=  pd.DataFrame(bigdisk1sites)

Df.to_csv(f"/bigdisk2/new/dropbox/bigdisk1_{date_time}.csv",index=False,header=False)

Df=  pd.DataFrame(bigdisk2sites)

Df.to_csv(f"/bigdisk2/new/dropbox/bigdisk2_{date_time}.csv",index=False,header=False)



for cursite in newsites:
    listzipfiles = (os.listdir(os.path.join(dropboxfolder,cursite)))
    input(f'Press enter to continue to extract site {cursite}....')
    for zipf in listzipfiles:
        curzip = os.path.join(dropboxfolder,cursite,zipf)

        curdestdir = os.path.join(bigdisk2,'silentcities',cursite)
        os.makedirs(curdestdir,exist_ok=True)

        print(f"Extracting {curzip} into {curdestdir}...")
        
        z = ZipFile(curzip)
        z.extractall(curdestdir)