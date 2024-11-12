import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the combined features from the JSON file
json_file_path = '../../Nrk_Songs_data/Hindi_Songs_Features.json'
with open(json_file_path, 'r') as f:
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

# Calculate Evaluation Metrics
silhouette_avg = silhouette_score(X_combined, clusters)
inertia = kmeans.inertia_
print(f"Silhouette Score: {silhouette_avg:.2f}")
print(f"Inertia: {inertia:.2f}")

# Compute the distances of each point to each cluster center
distances_to_centroids = kmeans.transform(X_combined)

# Compute likelihoods based on reciprocals of distances
reciprocals = 1 / distances_to_centroids
likelihoods = reciprocals / reciprocals.sum(axis=1, keepdims=True)  # Normalize to sum to 1 for each song
likelihoods = np.round(likelihoods, 2)  # Round to two decimal places

# Add likelihoods to the songs data
for i, song_name in enumerate(song_names):
    songs_features[song_name] = {
        "features": songs_features[song_name],
        "likelihoods": likelihoods[i].tolist()  # Convert to list for JSON compatibility
    }

# Save the updated JSON data back to a file
output_file_path = 'Hindi_Merged_Songs_Features.json'
with open(output_file_path, 'w') as f:
    json.dump(songs_features, f, indent=4)

print(f"Updated JSON file saved to {output_file_path}")

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
