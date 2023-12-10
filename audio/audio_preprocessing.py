import numpy as np
import librosa
import utils

def rescale(audio_file):
    # Scale audio file between -1 and 1 
    return ((audio_file - min(audio_file)) / (max(audio_file) - min(audio_file)) * 2) - 1    

def resample(audio_file, sr=16000):
    # Resample audio file to 16kHz
    return librosa.resample(audio_file, orig_sr=44100, target_sr=sr)

def create_windows(audio_file, window_size, sr=16000):
    frame_length = int(window_size * sr/1000)
    x = [audio_file[i:i+frame_length] for i in range(0, len(audio_file), frame_length)]
    return np.array(x[:-1])

def apply_stft(windows,sr=16000):
    # We have 960 ms worth of non overlapping windows
    # Now we need to create the spectrogram for each window
    hop_length = int(0.01 * sr) # 10ms of hop
    win_length = int(0.025 * sr) # 25ms of samples
    return np.abs(librosa.stft(windows, n_fft=512, hop_length=hop_length, win_length=win_length, window='hamming'))
    
def mel_filterbank(power_spectrum, sr = 16000):
    mel_filterbanks = librosa.filters.mel(sr=sr, n_fft=512, n_mels=64,fmin=125, fmax=7500)
    mel_spec = np.dot(mel_filterbanks, power_spectrum.T)
    return librosa.power_to_db(mel_spec.T, ref=np.max)

def preprocess(audio_file):
    audio_file = utils.load_audio(audio_file)
    audio_file = rescale(audio_file)
    audio_file = resample(audio_file)
    #windows = create_windows(audio_file, window_size=950)
    #spectogram = apply_stft(windows)
    #mel_spec = mel_filterbank(spectogram)
    #mel_spec = mel_spec.reshape(96,64,1,mel_spec.shape[0])
    return audio_file


    