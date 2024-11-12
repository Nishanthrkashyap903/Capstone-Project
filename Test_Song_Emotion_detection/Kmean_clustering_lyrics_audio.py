import librosa
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Preprocessing_Feature_lyrics import get_text_features
from sklearn.decomposition import TruncatedSVD


# Function to extract features from a song
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    # Stack all the features into a single vector
    return np.hstack([mfcc, chroma, spectral_contrast, zcr, rms])

# Paths to audio files
english_songs_path = 'English_Songs'
english_songs = [os.path.join(english_songs_path, f) for f in os.listdir(english_songs_path) if f.endswith('.mp3')]

# Extract features for English songs
english_audio_features = np.array([extract_audio_features(song) for song in english_songs])

# and 'tfidf_matrix' is the result of TF-IDF for lyrics
tfidf_matrix=get_text_features()

# Reduce dimensionality of the TF-IDF matrix
# svd = TruncatedSVD(n_components=30, random_state=42)
# reduced_tfidf = svd.fit_transform(tfidf_matrix)

# Combine Audio Features and Lyrics Features:
combined_features = np.hstack([english_audio_features,tfidf_matrix])

print(combined_features.shape)  # Combined features of both audio and lyrics

# Apply K-Means clustering to combined features
kmeans_combined = KMeans(n_clusters=4, random_state=42)
kmeans_combined.fit(combined_features)

# Get the cluster labels
song_clusters_combined = kmeans_combined.labels_

# Perform PCA to reduce dimensions to 2D for visualization
pca_combined = PCA(n_components=2)
reduced_combined = pca_combined.fit_transform(combined_features)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_combined[:, 0], reduced_combined[:, 1], c=song_clusters_combined, cmap='viridis')
plt.title('Songs Clustering (Combined Audio + Lyrics)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()