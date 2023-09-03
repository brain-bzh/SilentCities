import os,sys,shutil
import pandas as pd 


csvfile = sys.argv[1]
Dftest = pd.read_csv(csvfile)
if len(Dftest)<=1:
    print(f"Removing {csvfile}")
    #os.remove(csvfile)