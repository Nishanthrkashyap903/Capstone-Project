import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the CSV file containing audio features
csv_file_path = 'English_Songs.csv'  # Ensure this matches the correct CSV file
data = pd.read_csv(csv_file_path)

# Convert all feature columns to floats explicitly, except the 'Song Name'
for col in data.columns[1:]:  # Skipping the first column which is 'Song Name'
    data[col] = data[col].astype(float)

# Drop any rows with NaN values that may have arisen from invalid conversions
data = data.dropna()

# Separate the features (X) from the song names
X_features = data.drop(columns=['Song Name']).values  # Only feature columns for clustering

# Apply KMeans clustering
n_clusters = 4  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_features)  # Clustering based on features

# Print each song's cluster without any emotion mapping
print("Song Name, Cluster:")
for i, row in data.iterrows():
    print(f"Song: {row['Song Name']}, Cluster: {row['Cluster']}")

# Perform PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X_features)  # Reduce the feature space for visualization

# Plot the PCA-reduced features with cluster labels
plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7)

plt.title('English Songs Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Adding gridlines within the graph
plt.grid(which='both', linestyle='--', linewidth=0.7)
plt.minorticks_on()

# Show color legend for clusters
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

# Show the plot
plt.show()