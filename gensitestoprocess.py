import os 
import pandas as pd 
b1sites = os.listdir('/bigdisk1/silentcities/')
b2sites = os.listdir('/bigdisk2/silentcities/')
metapath = '/bigdisk2/meta_silentcities/site/'
nfilesdiff = 0 
dayfiles = 6*6*24 #number of files per day if one minute every ten minutes, divided in segments of 10 seconds
ndaythreshold = 7
threshold = ndaythreshold*dayfiles
print(dayfiles)
with open(r'sitestoprocess', 'w') as fp:
    sitetoprocess = []
    for cursite in b1sites+b2sites:
        sitefile = os.path.join(metapath,f"{cursite}.csv")
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

                if resultsmodif < sitemodif:
                    #print(f"{sitefile} has been modified, checking number of files")

                    Df = pd.read_csv(sitefile)
                    nfiles = len(Df)


                    num_lines_results = sum(1 for line in open(resultsfile))
                    if (nfiles*6 - num_lines_results) > (threshold):
                        nmissingfiles = (6*nfiles ) - num_lines_results
                        #print(f"for site {cursite} CSV there are {nfiles} files, but in results there are {num_lines_results-1} lines, which is less than {6*nfiles + threshold}")
                        print(f"for site {cursite} CSV it seems like {nmissingfiles} 10s segments are missing, corresponding to {nmissingfiles//dayfiles} days")

                        sitetoprocess.append(cursite)
                        fp.write(f"{cursite}\n")
                    
                    