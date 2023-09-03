import os 
import pandas as pd 
import numpy as np
b1sites = os.listdir('/bigdisk1/silentcities/')
b2sites = os.listdir('/bigdisk2/silentcities/')
metapath = '/bigdisk1/meta_silentcities/results/'
dbfilespath = '/bigdisk1/meta_silentcities/dbfiles/'
nfilesdiff = 0 
dayfiles = 6*24 #number of files per day if one minute every ten minutes, divided in segments of 10 seconds
ndaythreshold = 5
#threshold = ndaythreshold*dayfiles
threshold = 0
with open(r'sitestoprocess_final', 'w') as fp:
    sitetoprocess = []
    for cursite in b1sites+b2sites:
        sitefile = os.path.join(dbfilespath,f"{cursite}.csv")
        resultsfile = os.path.join(metapath,f"results_{cursite}.csv")
        if not(os.path.isfile(resultsfile)):
            sitetoprocess.append(cursite)
            fp.write(f"{cursite}\n")
            print(f"{resultsfile} does not exist")
        else:
            if not(os.path.isfile(sitefile)):
                print(f"{sitefile} does not exist")
            else:
                sitemodif = os.path.getmtime(sitefile)
                resultsmodif = os.path.getmtime(resultsfile)

                #if resultsmodif < sitemodif:
                #print(f"{sitefile} , checking number of files")

                Df = pd.read_csv(sitefile)
                resDf = pd.read_csv(resultsfile)
                nfiles = len(Df)
                

                #print(Df['datetime'].iloc[0],Df['datetime'].iloc[-1])
                
                #print(resDf['datetime'].iloc[0],resDf['datetime'].iloc[-1])

                listdates = [l[:12] for l in resDf['datetime'].to_list()]
                uniquedates = len(np.unique(listdates))
                #print(uniquedates,nfiles)


                

                if (uniquedates - nfiles) > (threshold):
                    nmissingfiles = uniquedates- nfiles
                    #print(f"for site {cursite} CSV there are {nfiles} files, but in results there are {num_lines_results-1} lines, which is less than {6*nfiles + threshold}")
                    print(f"for site {cursite} CSV it seems like {nmissingfiles} files are missing, corresponding to {nmissingfiles//dayfiles} days")
                    
                    sitetoprocess.append(cursite)
                    fp.write(f"{cursite}\n")
                    
                    