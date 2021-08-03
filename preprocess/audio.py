# coding=UTF-8
import os
import subprocess
 
def ffmpeg_VideoToAudio(VideoPath,SavePath):

    files = os.listdir(VideoPath)
    files_list = [VideoPath + f for f in files if f.endswith(('.avi','.mp4'))]
    for file in files_list:
        wavName=os.path.splitext(file)[0].split('/')[-1]
        wavNameNew=SavePath + wavName
        wavnewpath = SavePath + wavName
        wavnew = '/media/dancezha/audios/' + wavName
        # extract the audio
        strcmd="ffmpeg -i " + file + " -ar 8000" + " -f wav " + wavNameNew + ".wav"+ " -y"
        print('%s succeed'% file)
        print('-'*10)
        subprocess.call(strcmd,shell=True)
        strcmd1="ffmpeg-normalize "+ wavNameNew + ".wav" + " -nt rms -o " + wavnewpath + ".wav"
        subprocess.call(strcmd1,shell=True)
        strcmd2 = "ffmpeg -i "+ wavnewpath + ".wav" + " -ac 1 " + wavnew + '.wav'
        subprocess.call(strcmd2,shell=True)

VideoPath='/media/dance/videos/'
SavePath='/media/dance/audios/'
ffmpeg_VideoToAudio(VideoPath,SavePath)
