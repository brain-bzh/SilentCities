import os,sys,shutil
firstdisk = sys.argv[1] 
seconddisk = sys.argv[2]

folderlist = os.listdir(firstdisk)
secondlist = os.listdir(seconddisk)

domove = False

for folder in folderlist:

    if folder in secondlist:
        print("Folder {} was found on the second disk".format(folder))
        if domove:
            print("Moving all subfolders and files of {} to {}".format(os.path.join(firstdisk,folder),os.path.join(seconddisk,folder)))
            listsub = os.listdir(os.path.join(firstdisk,folder))
            for subfold in listsub:

                if os.path.isdir(os.path.join(firstdisk,folder,subfold)):
                    print("moving subfolder {}".format(os.path.join(firstdisk,folder,subfold)))
                    #shutil.copytree(os.path.join(firstdisk,folder,subfold),os.path.join(seconddisk,folder,subfold))
                    shutil.move(os.path.join(firstdisk,folder,subfold),os.path.join(seconddisk,folder))
                    #print("removing subfolder {}".format(os.path.join(firstdisk,folder,subfold)))
                    #shutil.rmtree(os.path.join(firstdisk,folder,subfold))
                else:
                    print("moving file {}".format(os.path.join(firstdisk,folder,subfold)))
                    shutil.move(os.path.join(firstdisk,folder,subfold),os.path.join(seconddisk,folder))
            print("removing folder {}".format(os.path.join(firstdisk,folder)))
            shutil.rmtree(os.path.join(firstdisk,folder))