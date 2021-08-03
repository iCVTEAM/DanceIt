#!/usr/bin/env python

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import json
import os

def mfcc_detect(Path,SavePath,mark,data_k):

   (rate,sig) = wav.read(Path)
   print(Path)
   print(len(sig))
   print(rate)
   #print(sig)
   mfcc_feat = mfcc(sig,rate,winlen=0.08,winstep=0.04166667,numcep=13)
   mfcc_feat = mfcc_feat.reshape(1,-1)

   lenx = len(mfcc_feat[0])//13
   print(lenx)	
   for i in range(lenx):
      data_i = []
      for count in range(13):
         data_i.append(mfcc_feat[0][i*13 + count])
      data_k[mark + i] = data_i

   return lenx

def file_find(Path,SavePath):
   
   mark = 0
   t = 0
   data_k = {}
   for dir in os.listdir(Path):

      dir = os.path.join(Path,dir)
      lenx = mfcc_detect(dir,SavePath,mark,data_k)
      mark += lenx
      print('输出第 %d 个文件'%(t))
      t += 1

   with open(SavePath,'a') as f:
      json.dump(data_k,f)


if __name__ == '__main__':
    file_find('/media/dance/audios', '/home/dance/train_audio.json')
