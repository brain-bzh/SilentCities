import os 
import sys
basepath = sys.argv[1] 

with open('arbo.txt','w') as f:
    for curp in os.listdir(basepath):
        #print(curp)
        for cursub in os.listdir(os.path.join(basepath,curp)):
            if os.path.isdir(os.path.join(basepath,curp,cursub)):
                print(os.path.join(curp,cursub),file=f)
