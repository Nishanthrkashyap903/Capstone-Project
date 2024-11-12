import pandas as pd
import numpy as np
import librosa
import os

# Function to extract specified audio features from a song
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    # Extract short-term chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    aggregate_chroma = np.mean(chroma_stft)  # Single mean value for aggregate chroma

    # Other features
    rmse = np.mean(librosa.feature.rms(y=y).T, axis=0)[0]  # Single value for RMSE
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0]  # Single value
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)[0]  # Single value
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).T, axis=0)[0]  # Single value
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)[0]  # Single value

    # Calculate mean energy
    mean_energy = np.mean(librosa.feature.rms(y=y).T, axis=0)[0]  # Single mean energy value

    # Calculate mean of MFCCs and take a single mean value
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs)  # Single mean value for all MFCCs combined

    # Calculate Tonnetz features and take a single mean value
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz)  # Single mean value for all Tonnetz features

    # Stack all features into a single vector
    features = np.hstack([
        aggregate_chroma,
        rmse, 
        spectral_centroid, 
        spectral_bandwidth, 
        rolloff, 
        zero_crossing_rate,
        mean_energy,  # Single mean energy value
        mfcc_mean,    # Single mean MFCC value
        tonnetz_mean   # Single mean Tonnetz value
    ])

    return features

# Path to the English songs directory
english_songs_path = '../Songs_Data/English_Songs'
english_songs = [os.path.join(english_songs_path, f) for f in os.listdir(english_songs_path) if f.endswith('.mp3')]

# Create a DataFrame to store song names and features
features_list = []

# Extract features for each English song
for song in english_songs:
    features = extract_audio_features(song)
    song_name = os.path.basename(song)
    features_list.append([song_name] + features.tolist())

# Create a DataFrame with the correct number of columns
columns = ['Song Name', 'aggregate_chroma', 'rmse', 'spectral_centroid', 
           'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 
           'mean_energy', 'mfcc_mean', 'tonnetz_mean']  # Single mean for MFCC and Tonnetz
features_df = pd.DataFrame(features_list, columns=columns)

# Save the features to a CSV file
features_df.to_csv('English_Songs.csv', index=False)