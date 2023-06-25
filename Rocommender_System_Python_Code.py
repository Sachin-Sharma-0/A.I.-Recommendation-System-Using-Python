#Recommender system based on tags and keywords (Documents Dataset):

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('documents.csv')

# Create a TF-IDF vectorizer to convert documents into numerical vectors
tfidf_vectorizer = TfidfVectorizer()

# Compute the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# Compute the cosine similarity between each pair of documents
cosine_similarities = cosine_similarity(tfidf_matrix)

# Define a function to recommend documents based on tags and keywords
def recommend_documents(tags, keywords):
    # Find all documents that match at least one of the given tags
    tag_matches = df[df['tags'].apply(lambda x: any(tag in x for tag in tags))]
    
    # Compute the TF-IDF matrix for the tag-matched documents
    tfidf_matrix_tag_matches = tfidf_vectorizer.transform(tag_matches['text'])
    
    # Compute the cosine similarities between the tag-matched documents and the given keywords
    keyword_similarities = cosine_similarity(tfidf_matrix_tag_matches, tfidf_vectorizer.transform(keywords))
    
    # Compute the total similarity scores for each tag-matched document
    total_similarities = keyword_similarities.sum(axis=1) * cosine_similarities[tag_matches.index].sum(axis=1)
    
    # Sort the tag-matched documents by their total similarity scores and return the top recommendations
    recommendations = tag_matches.iloc[total_similarities.argsort()[::-1]].reset_index(drop=True)
    return recommendations[['title', 'url']]
