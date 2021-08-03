#### Introduction

This code provides an initial version for the implementation dance generation model
 of the TIP2021 paper "DanceIt: Music-inspired dancing video synthesis".
This project are still under construction.

#### Preparing environment

pip install -r requirement.txt

#### How to run
##### For training:
###### For [Tsinghua dance dataset](https://github.com/Music-to-dance-motion-synthesis/dataset)

1. Download the dataset and unzip them in your customized path.

2. Build the train_data.json and truth_data.json by running DataProcess.py.

###### For dance video dataset

1. Download the [dataset](https://drive.google.com/file/d/1mrlEMFIJfXpsSEM_LjkV0ylNuB2LMYRD/view?usp=sharing) and unzip them in your customized path. 

2. Use [OpenPose](https://github.com/YangZeyu95/unofficial-implement-of-openpose). to extract keypoints (from dance videos).

3. Run preprocess/audio.py to extract the audio of the videos.
   
4. Run preprocess/audiomfcc.py to extract audio features.

5. Run preprocess/CombineAudioDance.py to build the train_data.json and truth_data.json.

Run main.py for training.

###### For other ways:

Download the preprocessed [data](https://drive.google.com/file/d/1vPyOqaIT-nmB5Yb8HQ0FZk8Usg2RD8Vp/view?usp=sharing) to test.

##### For testing:

python main.py --audio_file audio_path --test_model checkpoint/best_model_db.pth

#### To do:

The project is still in the process of further optimization. The initial version was used to verify the validity of our method.

#### Citations:

Please remember to cite us if u find this useful.

@article{guo2021danceit,

  title={DanceIt: Music-inspired Dancing Video Synthesis},

  author={Guo, Xin and Zhao, Yifan and Li, Jia},

  journal={IEEE Transactions on Image Processing},

  year={2021},

  publisher={IEEE}

}

#### License

The code of the paper is freely available for non-commercial purposes. Permission is granted to use the code given that you agree:

1. That the code comes "AS IS", without express or implied warranty. The authors of the code do not accept any responsibility for errors or omissions.

2. That you include necessary references to the paper [1] in any work that makes use of the code. 

3. That you may not use the code or any derivative work for commercial purposes as, for example, licensing or selling the code, or using the code with a purpose to procure a commercial gain.

4. That you do not distribute this code or modified versions. 

5. That all rights not expressly granted to you are reserved by the authors of the code.

# Dance2Music
