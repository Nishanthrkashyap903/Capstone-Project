import json

# Paths to the files
metadata_file_path = '../../Nrk_Songs_data/hindi-metadata.txt'
json_file_path = '../../Nrk_Songs_data/Hindi_Songs_Features.json'

# Read the metadata file to extract song_name, artist, and genre
song_metadata = {}

with open(metadata_file_path, 'r', encoding='utf-8') as f:
    # Skip the header
    next(f)
    for line in f:
        # Split the line by commas and ignore empty lines or improperly formatted lines
        parts = line.strip().split(',')
        
        # Ensure the line contains exactly three parts
        if len(parts) == 3:
            song_name, artist, genre = parts
            song_metadata[song_name] = {
                'artist': artist.strip(),
                'genre': genre.strip()
            }
        else:
            print(f"Skipping invalid line: {line.strip()}")

# Load the existing JSON file containing song features and likelihoods
with open(json_file_path, 'r', encoding='utf-8') as f:
    songs_features = json.load(f)

# Update the JSON file with artist and genre information
for song_name, song_info in songs_features.items():
    if song_name in song_metadata:
        # Add the artist(s) and genre to the song entry
        song_info['artist'] = song_metadata[song_name]['artist']
        song_info['genre'] = song_metadata[song_name]['genre']
    else:
        # If a song name is not found in the metadata file, you can handle it (optional)
        print(f"Warning: Metadata not found for song '{song_name}'")

# Save the updated JSON file
output_file_path = 'Hindi_Merged_Songs_Features_with_Metadata.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(songs_features, f, indent=4)

print(f"Updated JSON file saved to {output_file_path}")