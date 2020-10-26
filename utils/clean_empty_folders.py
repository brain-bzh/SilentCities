import os,sys,shutil
basepath = sys.argv[1] 

folderlist = os.listdir(basepath)


for folder in folderlist:
    filelist = os.listdir(os.path.join(basepath,folder))
    if len(filelist)==0:
        print("folder {} is empty, removing".format(folder))
        shutil.rmtree(os.path.join(basepath,folder))