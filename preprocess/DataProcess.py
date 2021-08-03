import librosa
import numpy as np
import glob
import json
import os

# Used to extract tsinghua datasets (https://github.com/Music-to-dance-motion-synthesis/dataset)

def main(data_dir, target_dir):
    data_list = glob.glob(data_dir + '\\DANCE_*')

    data = {}
    num = 0
    for sub_data in data_list:
        print(sub_data)
        music = sub_data + "\\audio.mp3"
        config = sub_data + "\\config.json"
        skeleton = sub_data + "\\skeletons.json"

        with open(skeleton, 'r') as f:
            dance_data = json.load(f)
        dance_skeleton = np.array(dance_data["skeletons"])
        with open(config, 'r') as t:
            config_data = json.load(t)
        s_p = config_data["start_position"]

        y, sr = librosa.load(music, sr=48000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=1920)
        mfccs = np.transpose(mfccs)
        len_d = min(len(dance_skeleton), len(mfccs))

        if s_p + len_d >= len(mfccs):
            len_d = len(mfccs) - s_p - 1

        t_duration = 100

        t = 0

        tem_feature_audio = []
        tem_feature_dance = []
        for i in range(s_p, s_p+len_d):
            for j in range(24):
                tem_feature_audio.append(mfccs[i][j].tolist())
            for k in range(3):
                for j in range(23):
                    tem_feature_dance.append(dance_skeleton[i-s_p][j][k].tolist())
            t += 1
            if t % t_duration == 0:
                tem_data = []
                tem_ad = []
                for j in range(len(tem_feature_audio)):
                    tem_ad.append(tem_feature_audio[j])
                for j in range(len(tem_feature_dance)):
                    tem_ad.append(tem_feature_dance[j])
                tem_data.append(tem_ad)
                tem_data.append(0)
                data[num] = tem_data
                tem_feature_audio = []
                tem_feature_dance = []
                num += 1
        print(num)
        f.close()

    for sub_data in data_list:
        print(sub_data)
        music = sub_data + "\\audio.mp3"
        config = sub_data + "\\config.json"
        skeleton = sub_data + "\\skeletons.json"

        with open(skeleton, 'r') as f:
            dance_data = json.load(f)
        dance_skeleton = np.array(dance_data["skeletons"])
        with open(config, 'r') as t:
            config_data = json.load(t)
        s_p = config_data["start_position"]

        y, sr = librosa.load(music, sr=48000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=1920)
        mfccs = np.transpose(mfccs)
        len_d = min(len(dance_skeleton), len(mfccs))

        if s_p + len_d >= len(mfccs):
            len_d = len(mfccs) - s_p - 1

        t_duration = 100

        t = 0

        tem_feature_audio = []
        tem_feature_dance = []
        for i in range(s_p, s_p+len_d-t_duration*2):
            for j in range(24):
                tem_feature_audio.append(mfccs[i][j].tolist())
            for k in range(3):
                for j in range(23):
                    tem_feature_dance.append(dance_skeleton[i-s_p+t_duration*2][j][k].tolist())
            t += 1
            if t % t_duration == 0:
                tem_data = []
                tem_ad = []
                for j in range(len(tem_feature_audio)):
                    tem_ad.append(tem_feature_audio[j])
                for j in range(len(tem_feature_dance)):
                    tem_ad.append(tem_feature_dance[j])
                tem_data.append(tem_ad)
                tem_data.append(1)
                data[num] = tem_data
                tem_feature_audio = []
                tem_feature_dance = []
                num += 1
        print(num)
        f.close()
    
    with open(target_dir + "\\train_data.json", "w") as f:
        json.dump(data, f)

if __name__ == '__main__':
    main("D:\\project\\dance2music\\dataset", "D:\\project\\dance2music\\code\\match")
