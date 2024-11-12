import os
import csv

# Folder containing the Hindi lyrics files
lyrics_folder = 'Hindi_lyrics'
csv_file = 'hindi_lyrics.csv'

# Open or create the CSV file and set up the header if it's not already present
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write header only if the file is empty
    if os.stat(csv_file).st_size == 0:
        writer.writerow(['Song_name', 'Lyrics'])
    
    # Iterate over each file in the Hindi_lyrics folder
    for filename in os.listdir(lyrics_folder):
        if filename.endswith('.txt'):  # Ensure we're only reading text files
            file_path = os.path.join(lyrics_folder, filename)
            
            # Read the lyrics from the file
            with open(file_path, 'r', encoding='utf-8') as lyrics_file:
                lyrics = lyrics_file.read()
                
            # Write the song name (filename without extension) and lyrics to the CSV
            song_name = os.path.splitext(filename)[0]
            writer.writerow([song_name, lyrics])

print("Lyrics have been successfully appended to hindi_lyrics.csv.")
