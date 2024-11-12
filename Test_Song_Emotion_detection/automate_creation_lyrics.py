import os

# Directory and file paths
metadata_file = "metadata.txt"
output_dir = "Hindi_lyrics"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the metadata file and process each song
with open(metadata_file, "r") as file:
    # Skip the header line
    next(file)
    
    for line in file:
        # Parse each line into song_name, artist, and genre
        song_info = line.strip().split(", ")
        song_name = song_info[0]
        
        # Define the path for the new file
        file_path = os.path.join(output_dir, f"{song_name}.txt")
        
        # Create the file and write an empty content or any default content
        with open(file_path, "w") as song_file:
            song_file.write("")  # Or add default content if needed

print("Files created successfully in the Hindi_lyrics directory.")
