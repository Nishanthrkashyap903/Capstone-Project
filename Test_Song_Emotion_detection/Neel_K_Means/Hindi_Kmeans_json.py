import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the combined features from the JSON file
with open('Hindi_Songs_Features.json', 'r') as f:
    songs_features = json.load(f)

# Extract song names and their features
song_names = list(songs_features.keys())
X_combined = np.array(list(songs_features.values()))

# Define and fit the KMeans model
n_clusters = 4  # Adjust as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_combined)

# Get the cluster labels
clusters = kmeans.labels_

# Perform PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_combined)

# Print the song name, x and y coordinates (PC1 and PC2), and cluster number for each song
print("Song Name, PC1 (x), PC2 (y), Cluster:")
for i, (x, y) in enumerate(reduced_data):
    print(f"Song: {song_names[i]}, PC1: {x:.2f}, PC2: {y:.2f}, Cluster: {clusters[i]}")

# Plot with cluster numbers
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')

# Annotate points with cluster numbers
for i, txt in enumerate(clusters):
    plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=9, color='black')

plt.title('Hindi Songs Clustering with Cluster Numbers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Cluster')
plt.show()
