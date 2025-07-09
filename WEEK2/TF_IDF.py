import math

def term_freq(corpus):
    # Corupus should be a list of strings
    num_docs = len(corpus)
    if num_docs == 0:
        return []
    lst = [] 
    for strings in corpus:
        # Convert to lowercase for consistency
        strings = strings.replace(".", "").lower()
        lst.append(strings.split(" "))
    
    
    vocab = []
    for list in lst:
        for x in list:
            if x not in vocab:
                vocab.append(x)
    vocab.sort()  # Sort vocabulary for consistent order
    
    tf = {}
    for x in vocab:
        lst2 = []
        for list in lst:
            count = list.count(x)
            lst2.append(count / len(list))
        tf[x] = lst2
        
    array = []
    for i in range(len(corpus)):
        lst3 = []
        for word in tf:
            lst3.append(tf[word][i])
        array.append(lst3)
    return vocab, tf 


# corpus = ['data science is one of the most important fields of science',
#           'this is one of the best data science courses',
#           'data scientists analyze data' ]

corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

            
def inverse_doc_freq(corpus):
    # Corupus should be a list of strings
    num_docs = len(corpus)
    if num_docs == 0:
        return []
    lst = [] 
    for strings in corpus:
        strings = strings.replace(".", "").lower()
        lst.append(strings.split(" "))
    vocab = []
    for list in lst:
        for x in list:
            if x not in vocab:
                vocab.append(x)
    vocab.sort()  # Sort vocabulary for consistent order
    idf = {}
    for x in vocab:
        count = 0
        for list in lst:
            if x in list:
                count += 1
        idf[x] = math.log(num_docs / count) # As i am creating vocab direct from corpus and does not removing any words so no need to add 1 in denominator of log.
    return idf      


def tf_idf(corpus):
    vocab, tf = term_freq(corpus)
    idf = inverse_doc_freq(corpus)
    tf_idf = {}
    for word in tf:
        tf_idf[word] = [tf[word][i] * idf[word] for i in range(len(tf[word]))]
    array = []
    for i in range(len(corpus)):
        lst3 = []
        for word in tf_idf:
            lst3.append(tf_idf[word][i])
        array.append(lst3)
    return vocab, array 
    
vocab, tf_idf_result = tf_idf(corpus)
print(f"\n Vocabulary: \n {vocab}")
print("\n \nTF-IDF Result:\n", tf_idf_result)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vectorizer = TfidfVectorizer()
vectorizer2 = CountVectorizer()

count_vector = vectorizer.fit_transform(corpus)
count_vector2 = vectorizer2.fit(corpus)

print("\n \nTfidfVectorizer Vectorizer Result: \n", count_vector.toarray())
print("\n \nFeature Names:\n \n", vectorizer.get_feature_names_out())

print("Vocabulary_CountVectorizer:\n", count_vector2.vocabulary_)
vector2 = vectorizer2.transform(corpus)
print("\n \nCountVectorizer Result: \n",vector2.toarray())
