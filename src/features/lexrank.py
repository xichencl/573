import nltk
import numpy as np
from scipy.spatial.distance import cosine
import itertools
import re

stops = set(nltk.corpus.stopwords.words('english'))


def power_method(cos_matrix, error):
    """
    @param cos_matrix: a sentence-sentence cos_sim matrix
    @param error: error for convergence of eigenvectors
    @return: an np array of an eigen vector
    """
    v1 = np.zeros(len(cos_matrix))
    v1.fill(1.0/len(cos_matrix)) 
    delta = 1.0
#     idx = 0
    while delta >= error:
        v2 = np.dot(cos_matrix.T, v1)
#         print(str(idx)+'iteration, vector: ', v2)
        
        delta = np.linalg.norm(np.subtract(v2,v1))
#         print(str(idx)+'iteration, delta: ', delta)
#         idx+=1
        v1 = v2
#     print(v1)
    return v1

def get_lexrank_scores(an_event, tf_idf_dict, threshold, error):
    """
    @param an_event: {doc_id : [sents]}
    @param tf_idf_dict: a word to tf_idf dict
    @param threshold: threshold for edges between nodes >=0.1
    @param error: error for calculating the convergence of eigenvectors
    @return: the lexrank scores for each indexed sentence and a lookup table for sentences       
    """
    all_sentences = []
    for doc in an_event.values():
        if isinstance(doc, list):
            all_sentences += doc
        else:
            all_sentences += nltk.sent_tokenize(re.sub('\\n', ' ', doc))
            
    cos_matrix, sent2idx, degree = get_cosine_sim_matrix(all_sentences, tf_idf_dict, threshold)
    for i in degree:
        cos_matrix[i] /= degree[i]
    eigen_vector = power_method(cos_matrix, error)
    #test 
#     idx = 0 
#     for x in np.nditer(eigen_vector):
#         print('sent '+str(idx), ' score: '+str(x))
#         idx+=1
    #end test
    return eigen_vector, sent2idx


def get_cosine_sim_matrix(all_sentences, tf_idf_dict, threshold):
    #remove stop words and lower_case words
    """
    @param all_sentences: a list of sentences
    @param tf_idf_dict: a lookup table for tf_idf
    @param threshold: the threshold for inclusion in cosine matrix
    @return a n x n cosine matrix (n=length of all_sentences)   
    """
    n = len(all_sentences)
    cos_matrix = np.zeros(shape=(n, n), dtype=float)
    degree = {}
    
    sent2idx = {}
    idx2sent = []
    idx = 0
    for s1, s2 in itertools.combinations_with_replacement(all_sentences, 2):
        if s1 not in sent2idx:
            sent2idx[s1] = idx
            idx2sent.append(s1)
            idx+=1
            
        if s2 not in sent2idx:
            sent2idx[s2] = idx
            idx2sent.append(s2)
            idx+=1
        # should make keys lower case previously 
        s1_tf_idf = np.array([tf_idf_dict.get(w1, 0) for w1 in s1.split() if w1.lower() not in stops])
        s2_tf_idf = np.array([tf_idf_dict.get(w2, 0) for w2 in s2.split() if w2.lower() not in stops])
        
        # test
#         print(s1_tf_idf)
#         print(s2_tf_idf)
        #padding
        if s1_tf_idf.size > s2_tf_idf.size:
            s2_tf_idf = np.append(s2_tf_idf, [0.0]*(s1_tf_idf.size-s2_tf_idf.size))
            
        elif s1_tf_idf.size < s2_tf_idf.size:
            s1_tf_idf = np.append(s1_tf_idf, [0.0]*(s2_tf_idf.size-s1_tf_idf.size))
        
        cos_sim = 1-cosine(s1_tf_idf, s2_tf_idf)
        
        if cos_sim>threshold:
            s1_idx = sent2idx[s1]
            cos_matrix[s1_idx, sent2idx[s2]] = 1.0
            if s1_idx not in degree:
                degree[s1_idx] = 0
            degree[s1_idx]+=1
#         else:
#             cos_matrix[sent2idx[s1], sent2idx[s2]] = 0
    
#     print('cluster', cos_matrix)
    return cos_matrix, sent2idx, degree
 


