import json
import pymysql

# Load the JSON data
with open('English_Merged_Songs_Features.json', 'r') as f:
    songs_data = json.load(f)

# MySQL database connection details
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Nishanthsep#3',
    'database': 'Music_Recomendation_Db'
}

# Emotion labels corresponding to likelihood array positions
emotion_labels = [
    "Positive High Arousal", 
    "Positive Low Arousal", 
    "Negative Low Arousal", 
    "Negative High Arousal"
]

# Establish MySQL connection
connection = pymysql.connect(**db_config)
cursor = connection.cursor()

# Create the 'songs' table if not exists
create_table_query = """
CREATE TABLE IF NOT EXISTS songs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    song_name VARCHAR(255),
    language VARCHAR(50),
    emotion_positive_high_arousal FLOAT,
    emotion_positive_low_arousal FLOAT,
    emotion_negative_low_arousal FLOAT,
    emotion_negative_high_arousal FLOAT
)
"""
cursor.execute(create_table_query)

# Insert song data into the songs table
for song_name, song_info in songs_data.items():
    likelihoods = song_info.get("likelihoods", [])
    # Assuming the likelihoods are in the correct order as per emotion_labels
    if len(likelihoods) == 4:
        song_name = song_name.strip()
        language = "English"  # As per your use case, adjust if needed

        # Prepare data for insert
        insert_query = """
        INSERT INTO songs (song_name, language, emotion_positive_high_arousal, emotion_positive_low_arousal, 
                           emotion_negative_high_arousal, emotion_negative_low_arousal)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            song_name, 
            language, 
            likelihoods[1],  # Positive High Arousal
            likelihoods[0],  # Positive Low Arousal
            likelihoods[2],  # Negative High Arousal
            likelihoods[3]   # Negative Low Arousal
        ))

# Commit the transaction
connection.commit()

# Close the connection
cursor.close()
connection.close()

print("Data inserted successfully.")
