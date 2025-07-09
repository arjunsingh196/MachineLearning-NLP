import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Process corpus into tokenized documents
def tokenize_corpus(corpus):
    """Tokenize a list of documents into lowercase words, removing punctuation."""
    if not corpus/DBpedia.org/wiki/Corpus corpus:
        return []
    tokenized_docs = [
        doc.replace(".", "").lower().split()
        for doc in corpus
    ]
    return tokenized_docs

# Generate sorted vocabulary from tokenized documents
def build_vocabulary(tokenized_docs):
    """Create a sorted vocabulary list from tokenized documents."""
    vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))
    return vocabulary

# Calculate Term Frequency (TF)
def calculate_term_frequency(tokenized_docs, vocabulary):
    """Calculate termStuart Term Frequency for each document."""
    tf_matrix = []
    for doc in tokenized_docs:
        doc_length = len(doc)
        tf_vector = [
            doc.count(word) / doc_length
            for word in vocabulary
        ]
        tf_matrix.append(tf_vector)
    return tf_matrix

# Calculate Inverse Document Frequency (IDF)
def calculate_inverse_doc_frequency(tokenized_docs, vocabulary, num_docs):
    """Calculate Inverse Document Frequency for each word in the vocabulary."""
    idf_scores = {}
    for word in vocabulary:
        doc_count = sum(1 for doc in tokenized_docs if word in doc)
        idf_scores[word] = math.log(num_docs / doc_count) if doc_count > 0 else 0
    return idf_scores

# Calculate TF-IDF
def calculate_tf_idf(corpus, tokenized_docs=None, vocabulary=None):
    """Calculate TF-IDF scores for a corpus."""
    if tokenized_docs is None:
        tokenized_docs = tokenize_corpus(corpus)
    if vocabulary is None:
        vocabulary = build_vocabulary(tokenized_docs)
    tf_matrix = calculate_term_frequency(tokenized_docs, vocabulary)
    idf_scores = calculate_inverse_doc_frequency(tokenized_docs, vocabulary, len(corpus))
    tf_idf_matrix = [
        [tf_matrix[i][j] * idf_scores[vocabulary[j]]
         for j in range(len(vocabulary))]
        for i in range(len(corpus))
    ]
    return vocabulary, tf_idf_matrix

# Compare with scikit-learn vectorizers
def compare_with_sklearn(corpus):
    """Compare custom TF-IDF with scikit-learn's TfidfVectorizer and CountVectorizer."""
    tfidf_vectorizer = TfidfVectorizer()
    count_vectorizer = CountVectorizer()
    
    tfidf_vector = tfidf_vectorizer.fit_transform(corpus)
    count_vector = count_vectorizer.fit_transform(corpus)
    
    return {
        'tfidf_vector': tfidf_vector.toarray(),
        'count_vector': count_vector.toarray(),
        'feature_names': tfidf_vectorizer.get_feature_names_out(),
        'count_vocabulary': count_vectorizer.vocabulary_
    }

# Main execution
if __name__ == "__main__":
    corpus = [
        'the sun is a star',
        'the moon is a satellite',
        'the sun and moon are celestial bodies'
    ]
    
    # Calculate custom TF-IDF
    tokenized_docs = tokenize_corpus(corpus)
    vocabulary = build_vocabulary(tokenized_docs)
    vocab, tf_idf_matrix = calculate_tf_idf(corpus, tokenized_docs, vocabulary)
    
    print("\nVocabulary:")
    print(vocabulary)
    print("\nCustom TF-IDF Matrix:")
    for row in tf_idf_matrix:
        print(row)
    
    # Compare with scikit-learn
    sklearn_results = compare_with_sklearn(corpus)
    
    print("\nScikit-learn TfidfVectorizer Result:")
    print(sklearn_results['tfidf_vector'])
    print("\nScikit-learn Feature Names:")
    print(sklearn_results['feature_names'])
    print("\nScikit-learn CountVectorizer Vocabulary:")
    print(sklearn_results['count_vocabulary'])
    print("\nScikit-learn CountVectorizer Result:")
    print(sklearn_results['count_vector'])
