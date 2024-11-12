from sklearn.feature_extraction.text import TfidfVectorizer
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the stopwords resource
nltk.download('stopwords')

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')  # For tokenization, if needed

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess the lyrics
def preprocess_lyrics(lyrics):
    # Remove non-alphabetic characters
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    # Convert to lowercase
    lyrics = lyrics.lower()
    # Tokenize
    tokens = lyrics.split()
    # Remove stopwords and apply stemming
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_text_features():

    # Directory containing all the lyrics text files
    lyrics_dir = 'lyrics'

    # List to store the contents of all lyrics files
    lyrics_list = []

    # Loop through all the files in the 'lyrics' directory
    for filename in os.listdir(lyrics_dir):
        if filename.endswith('.txt'):  # Check if the file is a .txt file
            file_path = os.path.join(lyrics_dir, filename)

            # Open the file and read its contents
            with open(file_path, 'r', encoding='utf-8') as file:
                lyrics = file.read()
                lyrics_list.append(lyrics)

    # Text Feature Extraction (TF-IDF or Word Embeddings)

    # Preprocess the lyrics
    preprocessed_lyrics = [preprocess_lyrics(lyrics) for lyrics in lyrics_list]

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=500)  # Limit to 500 most important words

    # Transform the lyrics into TF-IDF features
    tfidf_matrix = tfidf.fit_transform(preprocessed_lyrics).toarray()

    # print(tfidf_matrix.shape)  # Each row represents a song, each column represents a word feature
    return tfidf_matrix