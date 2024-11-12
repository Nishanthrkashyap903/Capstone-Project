from indicnlp import common
from indicnlp import loader

common.set_resources_path('./indic_nlp_resources')
loader.load()

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

with open('hindi_stopwords.txt', 'r', encoding='utf-8') as f:
    hindi_stopwords = set(f.read().splitlines())

# print(hindi_stopwords)

# Initialize normalizer
normalizer = IndicNormalizerFactory().get_normalizer("hi")

# Load your data
data = pd.read_csv('hindi_lyrics.csv') 

vectorizer = CountVectorizer(ngram_range=(1, 2),max_df=0.6, min_df=0.01)

X = vectorizer.fit_transform(data['Processed_Lyrics'])
# X = vectorizer.fit_transform(data['Lyrics'])

# print("X: ",X)

# Define and fit the LDA model
n_topics = 4  # Adjust the number of topics as needed
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# Get the vocabulary
vocab = vectorizer.get_feature_names_out()

# Display the top words for each topic for manual inspection
def display_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}: ", end='')
        top_words = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics.append(top_words)
        print(top_words)
    return topics

topics = display_topics(lda, vocab, n_top_words=10)

# Assign each song to a topic based on the highest probability
song_topics = []
topic_distributions = lda.transform(X)
for i, topic_dist in enumerate(topic_distributions):
    dominant_topic = topic_dist.argmax()
    song_topics.append((data['Song_name'][i], dominant_topic))

# Print each song name with its dominant topic
print("\nSong and Assigned Topic:")
for song_name, topic in song_topics:
    print(f"{song_name}: Topic {topic}")

# Use PCA to reduce topic distributions to 2D for visualization
pca = PCA(n_components=2)
topic_components_2d = pca.fit_transform(lda.components_)

# Plot the topics in 2D space
plt.figure(figsize=(10, 6))
plt.scatter(topic_components_2d[:, 0], topic_components_2d[:, 1], color='blue')

# Annotate each point with the topic number and sample words
for i, (x, y) in enumerate(topic_components_2d):
    plt.text(x, y, f"Topic {i}\n{topics[i]}", ha='center', fontsize=10, color='red')

# Set plot labels and title
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Topics in 2D Semantic Space (Manual Interpretation)')
plt.grid(True)
plt.show()