import json
import numpy as np
import collections

with open('/home/dancedata/train_keyps.json','r+') as f:
   data_keyps1 = json.load(f,object_pairs_hook=collections.OrderedDict)

with open('/home/dancedata/train_audio.json','r+') as t:
   data_audio1 = json.load(t,object_pairs_hook=collections.OrderedDict)

human_k = {}
keypoint = []
keypoint1 = []
keypoint2 = []
i = 0
maxlen = 0
t = 0

data_keyps = []
data_audio = []
data_3 = []

for (ind_1,audio_data),(ind,keyps_data) in zip(data_audio1.items(),data_keyps1.items()):
   audio_data = np.array(audio_data)
   keyps_data = np.array(keyps_data)
   if len(keyps_data) > maxlen: maxlen = len(keyps_data)
   if len(keyps_data) == 36:
      i += 1
      data_1 = []
      data_2 = []
      #if i <= 20000:continue
      #if i == 4000:break
      keyps = np.split(keyps_data,18)
      for key in keyps:
         data_1.append(key[0])
         data_2.append(key[1])
      data_1 += data_2
      data_keyps.append(data_1)
      data_3.append(audio_data)
      #data_1 = data_1.view(-1)
      #keyps_data = keyps_data.tolist()
      if i%24 == 0:
          data_keyps = np.array(data_keyps)
          data_keyps = data_keyps.flatten()
          data_3 = np.array(data_3)
          data_3 = data_3.flatten()
          data_keyps = data_keyps.tolist()
          data_3 = data_3.tolist()
          data_3.extend(data_keyps)
          data_3 = np.array(data_3)  
          data_3 = data_3.flatten()
          data_3 = data_3.tolist() 
          keypoint.append(data_3)
          keypoint1.append(1)
          data_3 = []
          data_keyps = []

keypoint2.append(keypoint)
keypoint2.append(keypoint1)
print(type(keypoint))
human_k[0] = keypoint2
json.dumps(human_k)
print('*'*10)
print(i)
print(maxlen)

# train_data.json and truth_data.json are the same format.
with open('/home/dancedata/train_data.json','a') as t:
   json.dump(human_k,t)

