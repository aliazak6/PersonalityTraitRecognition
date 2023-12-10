import librosa
import ffmpeg as ff
import numpy as np
import pickle 
import os
import audio_preprocessing
def load_audio(file_name, sr=44100):
    # https://librosa.org/doc/main/generated/librosa.load.html
    # This works with video files if you have ffmpeg installed
    #return librosa.load(file_name, sr=sr)[0]

    #https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md
    inputfile = ff.input(file_name)
    out = inputfile.output('-', format='f32le', acodec='pcm_f32le', ac=1, ar=sr,t=5)
    raw = out.run(capture_stdout=True)
    del inputfile, out
    return np.frombuffer(raw[0],np.float32)

def load_labels(annotations,file_name):
    labels = ['extraversion','agreeableness','conscientiousness','neuroticism','openness']
    ground_truth = np.array([annotations[trait][file_name] for trait in labels]).reshape(1,5)
    return ground_truth
def get_training_data(folder_path,annotation_path):
    with open(annotation_path, 'rb') as f:
        annotations = pickle.load(f,encoding='latin1')
    training_set_data = []  
    for filename in os.listdir(folder_path):
        filePath = folder_path+'/'+filename
        audio_data = audio_preprocessing.preprocess(filePath)
        if len(audio_data) != 5*16000:
            print('Error in file: ',filename,'. Delete it')
        else:
            training_set_data.append((audio_preprocessing.preprocess(filePath),load_labels(annotations,filename)))
    with open('train_dat/'+folder_path.split('/')[-2]+'.dat', "wb") as f:
        pickle.dump(training_set_data, f)

def get_validation_data(folder_path,annotation_path):
    with open(annotation_path, 'rb') as f:
        annotations = pickle.load(f,encoding='latin1')
    validation_set_data = []  
    for filename in os.listdir(folder_path):
        filePath = folder_path+'/'+filename
        audio_data = audio_preprocessing.preprocess(filePath)
        if len(audio_data) != 5*16000:
            print('Error in file: ',filename,'. Delete it')
        else:
            validation_set_data.append((audio_preprocessing.preprocess(filePath),load_labels(annotations,filename)))
    with open('val_dat/'+folder_path.split('/')[-2]+'.dat', "wb") as f:
        pickle.dump(validation_set_data, f)
