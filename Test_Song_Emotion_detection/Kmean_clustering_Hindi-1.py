import librosa
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    
    # Stack all features into a single vector
    return np.hstack([mfcc, chroma, spectral_contrast, zcr, rms, tempo, tonnetz, spectral_flux])

# Path to your folder with Hindi songs
hindi_songs_path = 'Hindi_Songs'
hindi_songs = [os.path.join(hindi_songs_path, f) for f in os.listdir(hindi_songs_path) if f.endswith('.mp3')]

# Extract features for Hindi songs
hindi_features = np.array( [extract_audio_features(song) for song in hindi_songs] )

# Perform KMeans clustering on the reduced data
kmeans = KMeans(n_clusters=4, random_state=42)  # Assuming 4 clusters for 4 emotions
hindi_clusters = kmeans.fit_predict(hindi_features)

# Print song name, coordinates, and cluster number for each song
print("Song Name, PC1 (x), PC2 (y), Cluster:")
for i, (x, y) in enumerate(hindi_features):
    song_name = os.path.basename(hindi_songs[i])  # Get just the song file name
    cluster = hindi_clusters[i]
    print(f"Song: {song_name}, PC1: {x:.2f}, PC2: {y:.2f}, Cluster: {cluster}")

# Plot the clustered data
plt.figure(figsize=(10, 6))
plt.scatter(hindi_features[:, 0], hindi_features[:, 1], c=hindi_clusters, cmap='viridis')
plt.colorbar(label='Cluster')
plt.title('Hindi Songs Clustering with Isomap + KMeans')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# Annotate points with cluster numbers
for i, txt in enumerate(hindi_clusters):
    plt.annotate(txt, (hindi_features[i, 0], hindi_features[i, 1]), fontsize=9, color='black')

plt.show()