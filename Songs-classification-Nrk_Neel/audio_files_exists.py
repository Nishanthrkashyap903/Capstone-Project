import os

def check_audio_files_in_directory(metadata_file, songs_directory):
    # Open and read the metadata file
    with open(metadata_file, 'r') as file:
        # Read the first line for labels (this will help us skip it later)
        labels = file.readline().strip().split(',')
        labels = [label.strip() for label in labels] 
        print("labels: ",labels)
        
        # Loop through each line in the file (skipping the first line)
        for line in file:
            # Remove leading/trailing spaces and split by ','
            line = [item.strip() for item in line.strip().split(',')]
            
            # Get the song name, artist(s), and genre
            song_name = line[0]

            print(song_name)
            
            if not song_name:
                continue

            # Create the path by appending song_name to the directory
            audio_file_path = os.path.join(songs_directory, song_name + '.mp3')
            print(audio_file_path)
            
            # Check if the audio file exists in the directory
            if not os.path.exists(audio_file_path):
                print(f"File not found: {audio_file_path}")
            else:
                print("Found")

# Example usage
metadata_file = '../Nrk_Songs_data/english-metadata.txt'
songs_directory = '../Nrk_Songs_data/English'
check_audio_files_in_directory(metadata_file, songs_directory)
