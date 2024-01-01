import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English NLP model from spaCy
nlp = spacy.load('en_core_web_sm')

# Sample text for summarization
text = """
Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction 
between computers and humans through natural language. It enables computers to understand, interpret, and 
generate human-like text. One of the applications of NLP is text summarization.

Text summarization is the process of distilling the most important information from a source text to create 
a shortened version without losing the key points. There are two main approaches to text summarization: 
extractive and abstractive. In this project, we'll implement an extractive summarization technique.

We'll use spaCy for tokenization and stop word removal, and scikit-learn for computing the similarity matrix 
to identify the most important sentences. The final result will be a concise summary of the input text.
"""

# Function to perform extractive summarization
def summarize_text(text, num_sentences=5):
    # Tokenize and process the input text using spaCy
    doc = nlp(text)

    # Remove stop words and create a list of non-stop words
    non_stop_words = [token.text for token in doc if not token.is_stop]

    # Convert the list of non-stop words to a string for CountVectorizer
    processed_text = ' '.join(non_stop_words)

    # Use CountVectorizer to create a document-term matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([processed_text])

    # Compute the cosine similarity matrix
    cosine_similarities = cosine_similarity(X, X)

    # Get the sentence scores based on cosine similarity
    sentence_scores = cosine_similarities[0]

    # Get the indices of the top sentences
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]

    # Sort the indices for proper order in the summary
    top_indices.sort()

    # Generate the final summary
    summary = ' '.join([doc.sentences[i].text for i in top_indices])

    return summary

# Generate and print the summary
summary = summarize_text(text)
print("Original Text:")
print(text)
print("\nSummary:")
print(summary)
