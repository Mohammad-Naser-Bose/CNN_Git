import numpy as np
import pandas as pd
import os
import matplotlib as plt
import librosa
import pickle
import pydub
from pydub import AudioSegment
from itertools import product,permutations

audio_file = r"\\solid\oemcom\MN_2024\Songs\Instrumental\Classical\A Baroque Letter - Aaron Kenny.mp3"
noise_file = r"\\solid\oemcom\MN_2024\Noise\to_use\td70mphsm.wav"
audio = librosa.load(audio_file,sr=None)
noise = librosa.load(noise_file,sr=None)
gain_levels_training_db = [l for l in np.arange(.5,10.5,1)] + [l2 for l2 in np.arange(20.5,30.5,1)]
gain_levels_testing_db = [l for l in np.arange(10.5,20.5,1)] 






################################################
def modify_recording_gain(audio,noise,pair):
    gain_audio = pair[0] 
    gain_noise = pair[1] 
    audio_desired_gain_linear = 10 ** (gain_audio / 20)
    noise_desired_gain_linear = 10 ** (gain_noise / 20)


    adjusted_audio = audio[0] * audio_desired_gain_linear
    adjusted_noise = noise[0] * noise_desired_gain_linear

    return adjusted_audio, adjusted_noise
def generate_noisy_recording(audio,noise):
    # Duplicate noise file to cover the whole song
    repeat_factor = int(np.ceil(len(audio) / len(noise)))
    adjusted_noise = np.tile(noise, repeat_factor)[:len(audio)]

    noisy_recording = audio + adjusted_noise

    return(noisy_recording)
def modify_recording_gain_testing(audio,noise,audio_gain):
    noise_gain = 0
    audio_desired_gain_linear = 10 ** (audio_gain / 20)
    noise_desired_gain_linear = 10 ** (noise_gain / 20)


    adjusted_audio = audio[0] * audio_desired_gain_linear
    adjusted_noise = noise[0] * noise_desired_gain_linear

    return adjusted_audio, adjusted_noise
def modify_noise_gain_testing(audio,noise,noise_gain):
    audio_gain = 0
    audio_desired_gain_linear = 10 ** (audio_gain / 20)
    noise_desired_gain_linear = 10 ** (noise_gain / 20)


    adjusted_audio = audio[0] * audio_desired_gain_linear
    adjusted_noise = noise[0] * noise_desired_gain_linear

    return adjusted_audio, adjusted_noise


################################################
testing_recordings_dict={}
pairs=list(product(gain_levels_training_db, gain_levels_training_db))
perms=permutations(pairs)
counter=0
for perm in perms:
    for pair in pairs:
        modified_audio, modified_noise = modify_recording_gain(audio,noise,pair)
        noisy_recording = generate_noisy_recording(modified_audio, modified_noise)
        testing_recordings_dict[counter]=noisy_recording
        counter+=1
        print(counter)

stop=1

with open("testing_recordings_dict.pkl", "wb") as file:
    pickle.dump(testing_recordings_dict, file)


counter=0
testing_recordings_const_noise_dict={}
pairs=list(gain_levels_training_db)
for audio_gain in pairs:
    modified_audio, modified_noise = modify_recording_gain_testing(audio,noise,audio_gain)
    noisy_recording = generate_noisy_recording(modified_audio, modified_noise)
    testing_recordings_const_noise_dict[counter]=noisy_recording
    counter+=1
with open("testing_recordings_const_noise_dict.pkl", "wb") as file:
    pickle.dump(testing_recordings_const_noise_dict, file)

counter=0
testing_recordings_const_audio_dict={}
pairs=list(gain_levels_training_db)
for noise_gain in pairs:
    modified_audio, modified_noise = modify_noise_gain_testing(audio,noise,noise_gain)
    noisy_recording = generate_noisy_recording(modified_audio, modified_noise)
    testing_recordings_const_audio_dict[counter]=noisy_recording
    counter+=1
with open("testing_recordings_const_audio_dict.pkl", "wb") as file:
    pickle.dump(testing_recordings_const_audio_dict, file)