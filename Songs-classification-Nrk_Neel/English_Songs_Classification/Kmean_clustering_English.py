import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the combined features from the JSON file
json_file_path = 'English_Merged_Songs_Features.json'
with open(json_file_path, 'r') as f:
    songs_data = json.load(f)

# Extract song names, features, artists, and genres
song_names = list(songs_data.keys())

# print(songs_data.values())

for song_info in songs_data.values():
    print(song_info['features'])
    break

features = np.array([song_info['features'] for song_info in songs_data.values()])
artists = [song_info['artists'] for song_info in songs_data.values()]
genres = [song_info['genre'] for song_info in songs_data.values()]

# Define and fit the KMeans model
n_clusters = 4  # Adjust as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(features)

# Get the cluster labels
clusters = kmeans.labels_

# Calculate Evaluation Metrics
silhouette_avg = silhouette_score(features, clusters)
inertia = kmeans.inertia_
print(f"Silhouette Score: {silhouette_avg:.2f}")
print(f"Inertia: {inertia:.2f}")

# Compute the distances of each point to each cluster center
distances_to_centroids = kmeans.transform(features)

# Compute likelihoods based on reciprocals of distances
reciprocals = 1 / distances_to_centroids
likelihoods = reciprocals / reciprocals.sum(axis=1, keepdims=True)  # Normalize to sum to 1 for each song
likelihoods = np.round(likelihoods, 2)  # Round to two decimal places

# Add likelihoods to the songs data
for i, song_name in enumerate(song_names):
    songs_data[song_name]['likelihoods'] = likelihoods[i].tolist()  # Convert to list for JSON compatibility

# Save the updated JSON data back to a file
output_file_path = 'English_Merged_Songs_Features.json'
with open(output_file_path, 'w') as f:
    json.dump(songs_data, f, indent=4)

# Perform PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(features)

# Print the song name, x and y coordinates (PC1 and PC2), cluster, artist, genre, and likelihood for each cluster
print("Song Name, PC1 (x), PC2 (y), Cluster, Artists, Genre, Likelihoods:")
for i, (x, y) in enumerate(reduced_data):
    print(f"Song: {song_names[i]}, PC1: {x:.2f}, PC2: {y:.2f}, Cluster: {clusters[i]}, Artists: {artists[i]}, Genre: {genres[i]}, Likelihoods: {likelihoods[i]}")

# Plot with cluster numbers
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')

# Annotate points with cluster numbers
for i, txt in enumerate(clusters):
    plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=9, color='black')

plt.title('English Songs Clustering with Cluster Numbers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Cluster')
plt.show()
