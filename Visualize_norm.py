# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import cv2
import os
import numpy as np
import librosa
FFMPEG_LOC = "ffmpeg "

def getUpperWholeBodyLines():
    CocoPairs = [
        (0, 1), (3, 4), (4, 5), (5, 6), (2, 7), (7, 8),
        (8, 9), (10, 11), (12, 13), (13, 14), (14, 15),
        (2, 16), (16, 17), (17, 18), (19, 20), (3, 12),
        (15, 21), (6, 22)
    ]   # = 19
    CocoPairsRender = CocoPairs
    return CocoPairsRender

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [255, 0, 85], [255, 255, 0],
              [255, 0, 85], [0, 255, 170]]

pairs = [[0,1], [3,4], [4,5], [5,6], [2,7], [7,8], [8,9], [10,11], [12,13], [13,14], [14,15], [2,16],
         [16,17], [17,18], [19,20], [3,12], [15,21], [6,22]]

def drawBodyAndFingers(frame, dance_frame, height, height_t=0.57639349, tform=None, color=(0, 255, 0)):

    centers = {}
    for n in range(23):
        x = dance_frame[0][n]
        y = dance_frame[1][n]
        center = (int(x)+256, 256-int(y))
        centers[n] = center
        cv2.circle(frame, center, 3, CocoColors[n], thickness=3, lineType=8, shift=0)

    for order, pair in enumerate(pairs):
        cv2.line(frame, centers[pair[0]], centers[pair[1]], CocoColors[order], 3)

    neck = (int(centers[12][0] + (centers[3][0]-centers[12][0])/2), int(centers[12][1] + (centers[3][1] - centers[12][1])/2))
    cv2.line(frame, centers[2], neck, CocoColors[22], 3)
    nose = (int(centers[1][0] + (centers[0][0]-centers[1][0])/2), int(centers[1][1] + (centers[0][1] - centers[1][1])/2))
    cv2.line(frame, nose, neck, CocoColors[16], 3)
    lfeet = (int(centers[11][0] + (centers[10][0]-centers[11][0])/2), int(centers[11][1] + (centers[10][1] - centers[11][1])/2))
    cv2.line(frame, centers[9], lfeet, CocoColors[18], 3)
    lfeet = (int(centers[19][0] + (centers[20][0]-centers[19][0])/2), int(centers[19][1] + (centers[20][1] - centers[19][1])/2))
    cv2.line(frame, centers[18], lfeet, CocoColors[18], 3)

    return frame

def writeAudio(vid_loc, audio_loc):
    new_vid_loc = vid_loc.split(".mp4")[0] + "_audio.mp4"
    cmd = FFMPEG_LOC + " -loglevel panic -i " + vid_loc + " -i " + audio_loc
    cmd += " -c:v copy -c:a aac -strict experimental " + new_vid_loc
    os.system(cmd)
    return new_vid_loc

def videoFromImages(imgs, outfile, audio_path=None, fps=25):
    fourcc_format = cv2.VideoWriter_fourcc(*'MP4V')
    size = imgs[0].shape[1], imgs[0].shape[0]
    vid = cv2.VideoWriter(outfile, fourcc_format, fps, size)
    for img in imgs:
        vid.write(img)
    vid.release()
    if audio_path is not None:
        writeAudio(outfile, audio_path)
    return outfile

def align_beat(dance_skeleton, beats):

    move_list = []
    for t in range(1, dance_skeleton.shape[0]):
        move_dis = 0
        for i in range(0, dance_skeleton.shape[1]):
            for j in range(0, dance_skeleton.shape[2]):
                move_dis += abs(dance_skeleton[t][i][j] - dance_skeleton[t-1][i][j])
        move_list.append(move_dis)

    adj_move = np.zeros((len(move_list),))
    for i in range(2, len(move_list)):
        adj_move[i] = abs(move_list[i] - move_list[i-1])

    key_loc_temp = np.argwhere(adj_move[:] < 0.1)

    key_loc = []
    num = 0
    count = 0
    for t in range(2, len(key_loc_temp)-1):
        if key_loc_temp[t+1] - key_loc_temp[t] == 1:
            num += 1
            count += key_loc_temp[t]
        else:
            num += 1
            count += key_loc_temp[t]
            key_loc.append(int(count/num))
            num = 0
            count = 0

    start = 0
    for t in range(0, len(key_loc)-1):
        if t == 0:
            begin = 0
            end = key_loc[t] + (key_loc[t+1] - key_loc[t])/2

            if (key_loc[t+1] - key_loc[t]) % 2 == 1:
                end += 1
        elif t == len(key_loc)-1:
            begin = key_loc[t] - (key_loc[t] - key_loc[t-1])/2
            end = len(dance_skeleton) - 1

            if (key_loc[t] - key_loc[t-1]) % 2 == 1:
                begin -= 1
        else:
            begin = key_loc[t] - (key_loc[t] - key_loc[t-1])/2
            end = key_loc[t] + (key_loc[t+1] - key_loc[t])/2

            if (key_loc[t] - key_loc[t-1]) % 2 == 1:
                begin -= 1
            if (key_loc[t+1] - key_loc[t]) % 2 == 1:
                end += 1

        while start < len(beats)-1 and abs(key_loc[t] - beats[start]) > abs(key_loc[t] - beats[start+1]):
            start += 1

        left_end = int(beats[start]) - int(begin)
        right_end = int(end) - int(beats[start])

        if left_end <= 0 or right_end <= 0:
            continue

        dance_left = cv2.resize(dance_skeleton[int(begin):key_loc[t], :, :], (dance_skeleton.shape[1], left_end))
        dance_right = cv2.resize(dance_skeleton[key_loc[t]:int(end), :, :], (dance_skeleton.shape[1], right_end))

        dance_skeleton[int(begin):int(beats[start]), :, :] = dance_left
        dance_skeleton[int(beats[start]):int(end), :, :] = dance_right
    return dance_skeleton

def visualizeKeypoints(target, outfile, audio_path=None, args=None, img_size=512, fps=24):

    images = []
    truth_data = []

    height_t = 0.67639349  # Set the distance between the nose and the feet of the target person

    for ind, targkeyps in enumerate(target):
        keyps = np.split(targkeyps, args.len_seg)
        height = 0.0
        # Select the distance between the nose and the foot as the measurement value for different people
        '''
        for targetkeyps in keyps:
            if targetkeyps[0]!=0 and targetkeyps[18]!=0 and targetkeyps[10]!=0 and targetkeyps[28]!=0 and targetkeyps[13]!=0 and targetkeyps[31] !=0:
                height = max(height, targetkeyps[28]-targetkeyps[18])
                height = max(height, targetkeyps[31]-targetkeyps[18])

        if height == 0:  height = 0.5
        height1 = height_t / height # Appropriate enlargement or reduction ratio
        '''
        for t in range(len(keyps)):
            keyps[t] = np.split(keyps[t], 3)
            keyps[t] = np.array(keyps[t])

            '''
            x = keyps[t][0]
            y = keyps[t][1]

            x_min = min(keyps[t][0])
            x_max = max(keyps[t][0])
            y_min = min(keyps[t][1])
            y_max = max(keyps[t][1])

            for i in range(23):
                keyps[t][0][i] = 0.5*512 + (keyps[t][0][i] - (x_max+x_min)/2)*height1*2.0
                keyps[t][1][i] = 0.93*512 + (keyps[t][1][i] - y_max) * height1
            '''
            truth_data.append(keyps[t])

    # Time sequence smoothing

    time_step = 8 # Selected window size
    No_do = 10 # smooth threshold

    for t in range(len(truth_data)):
        if t < 12: continue
        flag = 0
        if t < len(truth_data)-time_step:
            for j in range(args.num_node):
                if flag == 1: break
                for k in range(3):
                    if flag == 1: break
                    if abs(truth_data[t][k][j] - truth_data[t-1][k][j]) > No_do:
                        for l in range(args.num_node):
                            for m in range(2):
                                coef_final = np.zeros((4,))
                                pid_final = 0
                                for temp_1 in range(t-17,t-12):
                                    node_x = [0,time_step-1]
                                    node_y = []
                                    for temp_3 in node_x:
                                        node_y.append(truth_data[temp_3+temp_1][m][l])
                                    coef = np.polyfit(node_x,node_y,1)
                                    poly_fit = np.poly1d(coef)
                                    y = []
                                    for temp_2 in range(temp_1,temp_1+time_step):
                                        y.append(truth_data[temp_2][m][l]-poly_fit(temp_2-temp_1))
                                    flag = False
                                    pid = 0
                                    for list_index in range(0,time_step):
                                        for list_index_1 in range(list_index,time_step):
                                            if y[list_index] == y[list_index_1]:
                                                temp_4 = 1
                                                flag = True
                                                while list_index_1+temp_4 < time_step:
                                                    if y[list_index + temp_4] != y[list_index_1 + temp_4]:
                                                        flag = False
                                                        break
                                                    temp_4 += 1
                                                if flag:
                                                    pid = list_index_1 - list_index
                                    if flag and pid > 4:
                                        node_x = range(0,pid+1)
                                        y_flag = []
                                        for temp_5 in node_x:
                                            y_flag.append(y[temp_5])
                                        coef = np.polyfit(node_x,y_flag,3)
                                        coef_final += coef
                                        pid_final = max(pid,pid_final) 
                                    else:
                                        node_x = range(0,time_step)
                                        coef = np.polyfit(node_x,y,3)
                                        coef_final += coef
                                coef_final = coef_final/5
                                node_x = [0,time_step-1]
                                y = []
                                for temp_3 in node_x:
                                    y.append(truth_data[t+temp_3-4][m][l])
                                coef_init = np.polyfit(node_x,y,1)
                                poly_fit_init = np.poly1d(coef_init)
                                poly_fit = np.poly1d(coef_final)
                                if pid != 0:
                                    for temp_3 in range(time_step):
                                        truth_data[t+temp_3-int(time_step/2)][m][l] = poly_fit_init(temp_3//pid) + poly_fit(temp_3//pid)
                                else:
                                    for temp_3 in range(time_step):
                                        truth_data[t + temp_3 - int(time_step/2)][m][l] = poly_fit_init(temp_3) + poly_fit(temp_3)
                        flag = 1

    y, sr = librosa.load(audio_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr) #提取音乐节拍

    # Choose a certain proportion of music beats for different dance styles
    '''
    index = range(0, len(beats), 2)
    beats = beats[index]
    '''

    beats = np.round(librosa.frames_to_time(beats, sr=sr) * 25)

    truth_data = np.array(truth_data)
    truth_data = align_beat(truth_data, beats)

    i = 0
    for t in range(len(truth_data)):
        img_y, img_x = (img_size, img_size)
        targetkeyps = truth_data[t]
        newImage = np.ones((img_y, img_x, 3), dtype=np.uint8)
        newImage *= 255
        newImage = drawBodyAndFingers(newImage, targetkeyps, height)
        i += 1
        images.append(newImage)

    if images:
        videoFromImages(images, outfile, audio_path, fps=fps)
