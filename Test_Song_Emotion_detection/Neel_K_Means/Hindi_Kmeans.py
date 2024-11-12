import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os  # Importing os for file name extraction

# Path to the CSV file containing preprocessed lyrics and audio features
csv_file_path = 'Hindi_Songs.csv'

# Load preprocessed data
data = pd.read_csv(csv_file_path)

# Vectorize the preprocessed lyrics
vectorizer = CountVectorizer(max_df=0.85, min_df=0.01)
X_text = vectorizer.fit_transform(data['Processed Lyrics']).toarray()

# Extract audio features into a separate array
audio_features = data[['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']].values

# Combine text and audio features
X_combined = np.hstack((X_text, audio_features))

# Define and fit the KMeans model
n_clusters = 4  # Adjust as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_combined)

# Predict clusters
data['Cluster'] = kmeans.labels_

# Perform PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_combined)

# Print the song name, x and y coordinates (PC1 and PC2), and cluster number for each song
print("Song Name, PC1 (x), PC2 (y), Cluster:")
for i, (x, y) in enumerate(reduced_data):
    song_name = data['Song Name'].iloc[i]  # Get the song name from the DataFrame
    cluster = data['Cluster'].iloc[i]
    print(f"Song: {song_name}, PC1: {x:.2f}, PC2: {y:.2f}, Cluster: {cluster}")

# Plot with cluster numbers
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Cluster'], cmap='viridis')

# Annotate points with cluster numbers
for i, txt in enumerate(data['Cluster']):
    plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=9, color='black')

plt.title('Hindi Songs Clustering with Cluster Numbers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Cluster')
plt.show()