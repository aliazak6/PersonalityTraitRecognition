## This folder contains codes and materials needed to transform video files into embeddings and train audio model for personality trait prediction.
### utils.py -> 
```
This file is accessed by create_training_data.ipynb. It opens the video files, dumps the 16kHz mono sounds of first 5 seconds into *.dat file inside *_dat folders.  create_training_data.ipynb file takes the input files and outputs:
    x_train_embeddings.npy -> Has 5995 video's VGGish embeddings with size (1,640).
    Normally VGGish outputs (5,128) embeddings for 5 spectrogram. But we flatten it for ease in further use.
    We have excluded 5 video files from training dataset because they are shorter than 5 seconds and we took first 5 seconds of each video because of limited compute resources.
    x_train.npy -> It is just an intermediate file for creating embeddings. We had memory issues if we just run VGGish before restarting kernel.
    Name of the excluded training files:
        r93dLeVRk3U.003.mp4
        39o1zJFeM7E.004.mp4
        XpY-cxkbYdo.002.mp4
        Rp_gyvKE4hI.005.mp4
        r2HcJYjGK5s.003.mp4
    Same structure applies for x_val files except we used 1999 video files and excluded 1 file.
    Name of the excluded validation file:
        9uMpKla2OQM.003.mp4
```
    y_train.npy and y_val.npy are corresponding labels for video files. We just read annotation_*.pkl files and excluded job_interview part because we don't use that information anyway. So labels have 5995,5 and 1999,5 shapes respectively.

### audio_preprocessing.py -> 
```
This file includes the functions needed to create mel spectrograms, we realized that hub model doesn't need these steps because it inherently does all of them so they are actually idle. No need to use this file. We followed the steps in https://apple.github.io/turicreate/docs/userguide/sound_classifier/how-it-works.html for preprocessing.
```

### audio_train.ipynb -> 
```
This file runs in google colab due to limited resources we have. We have done several experiments with regression head and save the final weights for multi modal regression task.
```
