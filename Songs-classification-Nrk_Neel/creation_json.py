import json

def process_metadata(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        # Read the first line (labels) and strip spaces
        labels = file.readline().strip().split(',')
        labels = [label.strip() for label in labels]  # Remove extra spaces from labels

        # Process each subsequent line in the file
        for line in file:
            line = line.strip()  # Remove leading/trailing spaces
            if not line:  # Skip empty or blank lines
                continue

            # Split the line by commas and strip extra spaces from each part
            parts = [part.strip() for part in line.split(',')]
            
            if len(parts) != 3:
                print(f"Skipping invalid line: {line}")
                continue
            
            song_name, artist, genre = parts
            # Store artist and genre info for each song_name

            metadata[song_name] = {'artists': [ artist.strip() for artist in artist.split('and') ] , 'genre': genre}
    return metadata

def process_features(features_file):
    with open(features_file, 'r') as file:
        return json.load(file)

def create_combined_json(metadata_file, features_file, output_file):
    # Load metadata and features
    metadata = process_metadata(metadata_file)

    print("metadata keys: ",metadata.keys(),"\n")

    # print("paint the town red" in [key.lower() for key in metadata.keys()])s
    features = process_features(features_file)

    # print("features: ",features,"\n")

    combined_data = {}
    
    # Combine metadata and features based on song_name
    for song_name, song_features in features.items():
        print(song_name)
        if song_name in metadata.keys():
            print("Song name inside: ",song_name)
            combined_data[song_name] = {
                'features': song_features,
                'artists': metadata[song_name]['artists'],
                'genre': metadata[song_name]['genre']
            }
    print(len(combined_data.keys()))

    # Save the combined data to a new JSON file
    with open(output_file, 'w') as output:
        json.dump(combined_data, output, indent=4)
    print(f"Combined data has been written to {output_file}")

# Example usage
metadata_file = '../Neel_SongData/hindi-metadata.txt'  # Path to your metadata file
features_file = '../Neel_SongData/Hindi_Songs_Features.json'  # Path to your features file
output_file = 'Hindi_Songs_Classification/combined_data_neel.json'  # Path to the output file

create_combined_json(metadata_file, features_file, output_file)