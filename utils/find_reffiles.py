### the purpose of this file is to retrieve all the "ref" files for all sites, and copy them in a new folder.
### Inputs : 
# database_file : the database_pross.pkl file (output of metadata_site)
# site_csv_path : the path to all the csv for each site
# source_dir : One directory path with sites (as on the server there are two hard drives, this script has to be run twice )
# One directory path to store the copied ref files. site name will be appended to the file name
###

import os,shutil
import pandas as pd

database_file = '/bigdisk2/meta_silentcities/site/database_pross.pkl'
source_dir = '/bigdisk1/silentcities/'
target_dir = '/bigdisk2/meta_silentcities/ref_files/'

# Create the target dir (will be silent if existing)
os.makedirs(target_dir,exist_ok=True)

### Open the database

Df_allsites = pd.read_pickle(database_file)

### First we create a list of sites by ls'ing the source dir 

listsites = os.listdir(source_dir)

### We go through the list of directories / sites 

for cursite in listsites:
    # fetch the name of the ref file to find for this site
    ref_file = str(Df_allsites[Df_allsites['partID']==int(cursite)]['ref_file'].values)[2:-2]
    print(ref_file)

    ## Walk in folder to find it 

    finalpath = []
    for root, dirs, files in os.walk(os.path.join(source_dir,cursite), topdown=False):
        for name in files:
            if name == ref_file:
                finalpath.append(os.path.join(root,name))
                break

    if len(finalpath)>1:
        print("Found more than one matching ref file : ")
        print(finalpath)
    elif len(finalpath)==0:
        print("Ref file {} not found for site {} !!".format(ref_file,cursite))
    else:
        print("Final file found at {}".format(finalpath[0]))
        
        basepath = os.path.basename(finalpath[0])
        destfile = cursite + "_" + ref_file
        finalfilepath = os.path.join(target_dir,destfile)
        print("Copying...")
        print("Renaming to new name {}".format(finalfilepath))        
        shutil.copyfile(finalpath[0],finalfilepath)

