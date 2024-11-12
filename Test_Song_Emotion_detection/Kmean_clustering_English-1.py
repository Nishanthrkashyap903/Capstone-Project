"""Manual Approach-1"""

import librosa
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to extract enriched audio features from a song
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)

    # Additional features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Extract tempo
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)  # Tonnetz

    # Stack all features into a single vector
    features=np.hstack([mfcc, chroma, spectral_contrast, zcr, rms, spectral_centroid, tempo, tonnetz])

    return features

# print(extract_audio_features("sample_english_songs/Lover.mp3"))

# english_songs_path = 'sample_english_songs'
english_songs_path = 'English_Songs'

english_songs = [os.path.join(english_songs_path, f) for f in os.listdir(english_songs_path) if f.endswith('.mp3')]

# Extract features for English songs
english_features = [extract_audio_features(song) for song in english_songs]

from sklearn.cluster import KMeans

# Convert the list of features to a numpy array
X_english = np.array(english_features)

# Apply K-Means clustering to group songs
kmeans_english = KMeans(n_clusters=4, random_state=42)
kmeans_english.fit(X_english)

# Get the cluster labels (which songs belong to which cluster)
english_clusters = kmeans_english.labels_

# Assuming you already have the song paths in 'english_songs' and clusters in 'english_clusters'

# Create a dictionary to map songs to their cluster
english_song_clusters = {}
for idx, song in enumerate(english_songs):
    english_song_clusters[song] = english_clusters[idx]

pca_english = PCA(n_components=2)
reduced_english = pca_english.fit_transform(X_english)

# Print the song name, x and y coordinates (PC1 and PC2), and cluster number for each song
print("Song Name, PC1 (x), PC2 (y), Cluster:")
for i, (x, y) in enumerate(reduced_english):
    song_name = os.path.basename(english_songs[i])  # Get just the song file name
    cluster = english_clusters[i]
    print(f"Song: {song_name}, PC1: {x:.2f}, PC2: {y:.2f}, Cluster: {cluster}")

# Plot with cluster numbers
plt.figure(figsize=(10, 6))
plt.scatter(reduced_english[:, 0], reduced_english[:, 1], c=english_clusters, cmap='viridis')

# Annotate points with cluster numbers
for i, txt in enumerate(english_clusters):
    plt.annotate(txt, (reduced_english[i, 0], reduced_english[i, 1]), fontsize=9, color='black')

plt.title('English Songs Clustering with Cluster Numbers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

