import os
from librosa import get_duration

dirstodelete = []

filestodelete = []


for rootdir in ['/bigdisk2/silentcities','/bigdisk1/silentcities']:
	for root, dirs, files in os.walk(rootdir, topdown=False):
	    
	    #for curdir in dirs:
	        #if curdir[:7] == '__MACOS':
        #    print(f"Fake dir {os.path.join(root,curdir)}")
        #    dirstodelete.append(os.path.join(root,curdir))
            
            
    
	    for name in files:
	        if name[-3:].casefold() == 'wav':
	            curfile = os.path.join(root,name)
	            
	            ## filter the files that begin with ._
	            if name[:2] == '._':
	                print(f"Fake file {curfile}") 
	                filestodelete.append(curfile)                
	            ## Remove files that are less than 5k            
	            elif os.stat(curfile).st_size==0:
	                try:
	                    s=get_duration(curfile)
	                    print(s)
	                except:
	                    print(f"Empty File : {curfile}")
	                    filestodelete.append(curfile)
            


from tqdm import tqdm

print(f"Found {len(filestodelete)} empty files")

for curfiletodel in tqdm(filestodelete):
    try:
        os.remove(curfiletodel)
    except:
        pass
