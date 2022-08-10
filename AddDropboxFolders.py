#!/usr/bin/env python
# coding: utf-8

# In[1]:


from zipfile import ZipFile
import os


# testing the reading from a zip file and extracting just one file 

# In[3]:


z = ZipFile('/home/nfarrugi/bigdisk2/new/dropbox/SilentCitiesData/0069/0069_02.zip')
filelist = z.namelist()


# In[13]:


filelist


# Pipeline
# --
# 
# We will do this in the following steps : 
# 1. For each folder or dropbox, search for an existing folder on bigdisk1 or bigdisk2. 
# 2. If the folder is not founddd, this is a new site. 
# 3. Parse the list of new sites and put them on bigdisk2, extract them flat. Save the list of new sites as sites for which we will need to restart "metadata_file"
# 4. Parse all the other archives by checking their file list, compare with the csv of the corresponding site, copy if needed. If ANY copy is done, add the corresponding site to the list of sites to update. 
# 5. Relaunch metadata_file.py for all the sites that have been updated
# 6. Relaunch metadata_site.py 
# 7. Relaunch audio_processing.py for all the sites that have been updated

# 1. For each folder or dropbox, search for an existing folder on bigdisk1 or bigdisk2. 
# 2. If the folder is not found, this is a new site. 

# In[5]:


bigdisk1 = '/home/nfarrugi/bigdisk1'
bigdisk2 = '/home/nfarrugi/bigdisk2'

dropboxfolder = '/home/nfarrugi/bigdisk2/new/dropbox/SilentCitiesData/'


# In[10]:


newsites = []
bigdisk1sites = []
bigdisk2sites = []

for curdir in os.listdir(dropboxfolder):
    if os.path.isdir(os.path.join(bigdisk1,'silentcities',curdir)):
        print(f"{curdir} found on bigdisk1")
        bigdisk1sites.append(curdir)
    elif os.path.isdir(os.path.join(bigdisk2,'silentcities',curdir)):
        print(f"{curdir} found on bigdisk2")
        bigdisk2sites.append(curdir)
    else:
        newsites.append(curdir)
        print(f"{curdir} is a new site")


# 3. Parse the list of new sites and put them on bigdisk2, extract them flat. Save the list of new sites as sites for which we will need to restart "metadata_file"
# 

# In[11]:


def extract_list_from_zip(zipfile,destdir,filelist=None):
    ### filelist is a subset (potentially all) from the list of files in zipfile
    ### If not, take the whole list of files from the zip

    os.makedirs(destdir,exist_ok=True)

    z = ZipFile(zipfile)

    if filelist is None:
        filelist = z.namelist()
        print("Extracting all files...")
    
    print(f"Extracting {len(filelist)}files")

    for curfile in filelist:
        
        # In case in the filelist there are paths
        filenameonly = os.path.split(curfile)[1]

        # extract a specific file from the zip container
        f = z.open(curfile)

        # save the extracted file 
        content = f.read()
        f = open(os.path.join(destdir,filenameonly), 'wb')
        f.write(content)
        f.close()


# In[12]:


bigdisk1sites


# In[167]:


def list_all_files_from_dir(curfolder):
    tot = 0
    allfiles = []
    allzips = []
    allDf = []
    for root, dirs, files in os.walk(curfolder, topdown=False):
        for name in files:        
            if os.path.splitext(name)[1].casefold() == '.zip' and name[:2] != '._' and name[:2] != '__':
                try:
                    z = ZipFile(os.path.join(curfolder,name))
                    filelist = z.namelist()
                    ziplist = []
                    for f in filelist:
                        if f[0] != '_' and os.path.splitext(f)[1].casefold()=='.wav':
                            ziplist.append(os.path.split(f)[-1])
                            
                            
                    allfiles.append(ziplist)
                    allzips.append([os.path.join(curfolder,name)])
                    tot+=len(ziplist)
                    curDf = pd.DataFrame.from_dict({'filename':ziplist})
                    curDf['zip']=os.path.join(curfolder,name)
                    allDf.append(curDf)
                except Exception as e:
                    print(e)
                    print(os.path.join(curfolder,name))
    return tot,pd.concat(allDf,ignore_index=True)


# In[270]:


def extract_list_from_zip(zipfile,destdir,curfile):
    ### filelist is a subset (potentially all) from the list of files in zipfile
    ### If not, take the whole list of files from the zip
    os.makedirs(destdir,exist_ok=True)
    z = ZipFile(zipfile)
    # In case in the filelist there are paths
    filenameonly = os.path.split(curfile)[1]
    # extract a specific file from the zip container
    curfile_ = [x for x in z.namelist() if curfile in x][0]
    f = z.open(curfile_)
    # save the extracted file 
    content = f.read()
    f = open(os.path.join(destdir,filenameonly), 'wb')
    f.write(content)
    f.close()


# In[211]:


import pandas as pd
from tqdm import tqdm
tqdm.pandas()

for cursite in (bigdisk1sites):
    curfolder = os.path.join(dropboxfolder, cursite)
    tot,Dfdropbox = list_all_files_from_dir(curfolder)
    ## go find the CSV for this site 
    Df = pd.read_csv(f'/home/nfarrugi/bigdisk2/meta_silentcities/site/{cursite}.csv')
    df_ = Dfdropbox.join(Df[['filename','dB']].set_index('filename'),on='filename',how='left').copy()
    missing_df = df_[df_['dB'].isna()].copy()
    print(f"Site {cursite} : {len(Df)} files were already analyzed , {tot} files on dropbox, new {len(missing_df)} ")
    
    destdir_ = os.path.join(bigdisk1,'silentcities',cursite)
    #missing_df.apply(lambda x :extract_list_from_zip(x.zip,destdir_,x.filename),axis=1 )
    missing_df.progress_apply(lambda x :extract_list_from_zip(x.zip,destdir_,x.filename),axis=1 )


# In[275]:


for cursite in (bigdisk2sites):
    curfolder = os.path.join(dropboxfolder, cursite)
    tot,Dfdropbox = list_all_files_from_dir(curfolder)
    ## go find the CSV for this site 
    Df = pd.read_csv(f'/home/nfarrugi/bigdisk2/meta_silentcities/site/{cursite}.csv')
    df_ = Dfdropbox.join(Df[['filename','dB']].set_index('filename'),on='filename',how='left').copy()
    missing_df = df_[df_['dB'].isna()].copy()
    print(f"Site {cursite} : {len(Df)} files were already analyzed , {tot} files on dropbox, new {len(missing_df)} ")
    
    destdir_ = os.path.join(bigdisk2,'silentcities',cursite)
    missing_df.progress_apply(lambda x :extract_list_from_zip(x.zip,destdir_,x.filename),axis=1 )

