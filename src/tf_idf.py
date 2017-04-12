import nltk
import math

# return a dictionary with the tf*idfs for a set of documents
def get_tf_idfs(docs):
    tf_idfs = {}
    doc_counts = {}
    word_counts = {}

    for doc in docs.keys():
        a_doc = docs[doc]
        words = nltk.word_tokenize(' '.join(a_doc))
        for word in words:
            if word in word_counts.keys():
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        vocab = set(words)
        for term in vocab:
            if term in doc_counts.keys():
                doc_counts[term] += 1
            else:
                doc_counts[term] = 1
    for key in doc_counts.keys():
        tf_idfs[key] = word_counts[key] * math.log(float(len(docs)) / doc_counts[key])
    return tf_idfs

def get_tf_idf_average(sent, tf_idfs):
    tf_idf_sum = 0
    words = nltk.word_tokenize(sent)
    for word in words:
        if word in tf_idfs.keys():
            tf_idf_sum += tf_idfs[word]
    return float(tf_idf_sum) / len(words)