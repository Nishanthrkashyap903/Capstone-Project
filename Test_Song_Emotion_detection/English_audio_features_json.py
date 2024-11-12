import pandas as pd
import numpy as np
import librosa
import os

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    # Additional features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Extract tempo
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)  # Tonnetz

    # Stack all features into a single vector
    features=np.hstack([mfcc, chroma, spectral_contrast, zcr, rms, tempo, tonnetz])

    return features

# Path to the English songs directory
english_songs_path = '../Nrk_Songs_data/English'
english_songs = [os.path.join(english_songs_path, f) for f in os.listdir(english_songs_path) if f.endswith('.mp3')]

print(english_songs)


import json

# Save features to JSON
features_dict = {}
for song in english_songs:
    features = extract_audio_features(song).tolist()
    song_name = os.path.basename(song)
    features_dict[song_name] = features

# Write to JSON file
with open('English_Songs_Features.json', 'w') as f:
    json.dump(features_dict, f)