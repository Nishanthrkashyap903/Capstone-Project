from indicnlp import common
from indicnlp import loader

common.set_resources_path('./indic_nlp_resources')
loader.load()

import pandas as pd
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Load Hindi stopwords
with open('hindi_stopwords.txt', 'r', encoding='utf-8') as f:
    hindi_stopwords = set(f.read().splitlines())

# Initialize normalizer
normalizer = IndicNormalizerFactory().get_normalizer("hi")

# Load your data
data = pd.read_csv('hindi_lyrics.csv') 

# Custom Hindi stemmer function
# def custom_hindi_stemmer(word):
#     # Define common suffixes in Hindi
#     hindi_suffixes = [
#         'ा', 'ी', 'ें', 'ों', 'े', 'ि', 'ू', 'ु',      # Gender/Plural variations
#         'ता', 'ते', 'ती', 'ना', 'वाला',               # Verb tense/aspect
#         'वाले', 'वाली', 'कर', 'हुए', 'करना',         # Participle forms
#         'में', 'से', 'के', 'पर', 'तक', 'की', 'को'    # Postpositions
#     ]
    
#     # Remove suffix if found at the end of the word
#     for suffix in hindi_suffixes:
#         if word.endswith(suffix):
#             return word[:-len(suffix)]
#     return word

# Preprocess lyrics function
def preprocess_lyrics(lyrics):
    # Normalize the text
    normalized_text = normalizer.normalize(lyrics)
    
    # Tokenize, remove stopwords, and apply stemming
    tokens = normalized_text.split()
    processed_tokens = [word for word in tokens if word not in hindi_stopwords]
    
    return " ".join(processed_tokens)

# Apply preprocessing to lyrics
data['Processed_Lyrics'] = data['Lyrics'].apply(preprocess_lyrics)

# Save the DataFrame back to 'hindi_lyrics.csv', including the new "Processed_Lyrics" column
data.to_csv('hindi_lyrics.csv', index=False, encoding='utf-8')
