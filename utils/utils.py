## Author : Nicolas Farrugia, Feb/March 2020

import datetime
import pickle
import os

def convert_Audio(mediaFile, outFile):
    cmd = 'ffmpeg -i '+mediaFile+' '+outFile
    os.system(cmd)
    return outFile

def gen_srt(strlabel,onset,srtfile,duration=2,num=1):

    starttime = onset
    endtime = starttime + duration

    string_start = datetime.time(0,starttime//60,starttime%60).strftime("%H:%M:%S")

    string_end = datetime.time(0,endtime//60,endtime%60).strftime("%H:%M:%S")

    with open(srtfile,'a') as f:
        f.write("{}\n".format(num+1))
        f.write("{starttime} --> {endtime}\n".format(starttime=string_start,endtime=string_end))
        f.write("{}\n".format(strlabel))
        f.write("\n")



def read_audio_hdr(strWAVFile,verbose=False):
# Open file
    fileIn = open(strWAVFile, 'rb')
        
    # end try
    # Read in all data
    bufHeader = fileIn.read(200)
    fileIn.close()


    AMOhdr = str(bufHeader[56:170])
    
    
    words = AMOhdr.split()
    if 'Recorded' in words[0]:
        curtime = datetime.datetime.strptime(str(words[2]),"%H:%M:%S").time()
        curdate = datetime.datetime.strptime(str(words[3]),"%d/%m/%Y").date()
        curdate_time = datetime.datetime.strptime(str(words[3])+str(words[2]),"%d/%m/%Y%H:%M:%S")
        AMOid = words[7]
        if verbose:
            print('Audiomoth detected, ID is {}'.format(AMOid))
            print("Recording date and time is {}".format(curdate_time))
            
        #gain = float(words[11])
        #battery = words[16]

        metadata = dict(time=curtime,date=curdate,id=AMOid,datetime=curdate_time)
    else:
        strFile = os.path.basename(strWAVFile)
        strFile = strFile.split('_')
        curtime = datetime.datetime.strptime(strFile[-1][:-4],"%H%M%S").time()
        curdate = datetime.datetime.strptime(strFile[-2],"%Y%m%d").date()
        curdate_time = datetime.datetime.strptime('{}_{}'.format(strFile[-2],strFile[-1][:-4]),"%Y%m%d_%H%M%S")
        
        SM4id = strFile[0]
        if verbose:
            print('SM4 detected, ID is {}'.format(SM4id))
            print("Recording date and time is {}".format(curdate_time))


        metadata = dict(time=curtime,date=curdate,datetime=curdate_time,id=SM4id,gain=0)

    return AMOhdr,metadata


def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)