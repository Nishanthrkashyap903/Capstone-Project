import os
import pandas as pd
import librosa
import numpy as np
from indicnlp import common, loader
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import re

# Set up resources for Indic NLP
common.set_resources_path('../indic_nlp_resources')
loader.load()

# Load Hindi stopwords
with open('./hindi_stopwords.txt', 'r', encoding='utf-8') as f:
    hindi_stopwords = set(f.read().splitlines())

# Initialize normalizer for Hindi
normalizer = IndicNormalizerFactory().get_normalizer("hi")

# Define custom Hindi stemmer function
def custom_hindi_stemmer(word):
    hindi_suffixes = [
        'ा', 'ी', 'ें', 'ों', 'े', 'ि', 'ू', 'ु',
        'ता', 'ते', 'ती', 'ना', 'वाला', 
        'वाले', 'वाली', 'कर', 'हुए', 'करना',
        'में', 'से', 'के', 'पर', 'तक', 'की', 'को'
    ]
    for suffix in hindi_suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# Function to tokenize text manually
def manual_tokenize(text):
    # Basic tokenization by splitting on whitespace and normalizing whitespace
    tokens = re.split(r'\s+', text.strip())
    return [token for token in tokens if token]

# Function to create n-grams
def create_ngrams(tokens, n):
    return ['_'.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# Function to preprocess lyrics
def preprocess_lyrics(lyrics):
    # Normalize the text
    normalized_text = normalizer.normalize(lyrics)
    
    # Remove non-informative symbols and numbers
    normalized_text = re.sub(r'[^\w\s]', '', normalized_text)  # Remove punctuation
    normalized_text = re.sub(r'\d+', '', normalized_text)      # Remove numbers

    # Tokenize manually
    tokens = manual_tokenize(normalized_text)

    # Remove stopwords and apply custom stemming
    processed_tokens = [
        custom_hindi_stemmer(word) for word in tokens 
        if word not in hindi_stopwords and len(word) > 1
    ]

    # Create unigrams, bigrams, and trigrams
    unigrams = processed_tokens
    bigrams = create_ngrams(processed_tokens, 2)
    trigrams = create_ngrams(processed_tokens, 3)

    # Return a joined string of unigrams, bigrams, and trigrams
    return " ".join(unigrams + bigrams + trigrams)

# Path to the folder containing Hindi song lyrics text files
lyrics_folder = '../Songs_Data/Hindi_lyrics'  # Update with your folder path
csv_file_path = 'Hindi_Songs.csv'

# List to store song data
song_data = []

# Process each text file in the folder
for filename in os.listdir(lyrics_folder):
    if filename.endswith('.txt'):
        song_name = filename.split('.')[0]  # Song name without file extension
        file_path = os.path.join(lyrics_folder, filename)
        
        # Read the original lyrics
        with open(file_path, 'r', encoding='utf-8') as f:
            original_lyrics = f.read()
        
        # Preprocess the lyrics
        preprocessed_lyrics = preprocess_lyrics(original_lyrics)
        
        # Append song data to list
        song_data.append({
            'Song Name': song_name,
            'Original Lyrics': original_lyrics,
            'Processed Lyrics': preprocessed_lyrics
        })

# Convert song data to DataFrame and save to CSV
df = pd.DataFrame(song_data)
df.to_csv(csv_file_path, index=False, encoding='utf-8')

print(f"Data has been successfully saved to {csv_file_path}.")

# Define the path to your audio files and the existing CSV
audio_folder_path = "../Songs_Data/Hindi_Songs"  # Update with the correct path to your audio files
csv_file_path = "Hindi_Songs.csv"  # Update with the path to your CSV file

# Load the existing CSV file
df = pd.read_csv(csv_file_path)

# Initialize empty lists for each feature to be added as columns
features = {
    'chroma_stft': [],
    'rmse': [],
    'spectral_centroid': [],
    'spectral_bandwidth': [],
    'rolloff': [],
    'zero_crossing_rate': []
}

# Process each song file listed in the CSV
for idx, row in df.iterrows():
    # Construct the full file path by combining folder path and song name
    audio_file_path = os.path.join(audio_folder_path, f"{row['Song Name']}.mp3")
    
    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        print(f"Error: File not found - {audio_file_path}")
        # Append NaN values for missing files
        for feature in features:
            features[feature].append(np.nan)
        continue  # Skip to the next file if the current one is missing
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None)

        # Extract features and add to the corresponding lists
        features['chroma_stft'].append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        features['rmse'].append(np.mean(librosa.feature.rms(y=y)))
        features['spectral_centroid'].append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features['spectral_bandwidth'].append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features['rolloff'].append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features['zero_crossing_rate'].append(np.mean(librosa.feature.zero_crossing_rate(y)))
        
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        # Append NaN values for files that couldn't be processed
        for feature in features:
            features[feature].append(np.nan)

# Add the feature lists as new columns to the original DataFrame
for feature, values in features.items():
    df[feature] = values

# Save the updated DataFrame back to the same CSV file
df.to_csv(csv_file_path, index=False, encoding='utf-8')
print(f"Updated CSV file saved to {csv_file_path}.")
