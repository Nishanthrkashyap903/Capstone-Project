import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import json

# Path to the CSV file
csv_file_path = 'Hindi_Songs.csv'

# Load preprocessed data
data = pd.read_csv(csv_file_path)

# Vectorize the processed lyrics
vectorizer = CountVectorizer(max_df=0.85, min_df=0.01)
X_text = vectorizer.fit_transform(data['Processed Lyrics']).toarray()

# Extract audio features into a separate array
audio_features = data[['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']].values

# Combine text and audio features
X_combined = np.hstack((X_text, audio_features))

# Create a dictionary to store each song's name and combined features
songs_features = {}
for i, song_name in enumerate(data['Song Name']):
    songs_features[song_name] = X_combined[i].tolist()  # Convert array to list for JSON compatibility

# Save the combined features to a JSON file
with open('Hindi_Songs_Combined_Features.json', 'w') as f:
    json.dump(songs_features, f, indent=4)

print("JSON file 'Hindi_Songs_Combined_Features.json' created successfully.")
