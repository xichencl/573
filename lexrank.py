"""
return a vector of lexrank scores for each sentence in document cluster
main method: get_lexrank_scores()
"""


import nltk
import numpy as np
from scipy.spatial.distance import cosine
import itertools
import re
import math
import json

def test_lexrank(file):
    all_docs = json.load(open(file, 'r'))
    tf_idf_dict = {}
    for topic in all_docs:
        tf_idf_dict[topic] = {}
        tf_idf_dict[topic]['tf_idf']=get_tf_idfs(all_docs[topic])
#         print(tf_idf_dict[topic]['tf_idf'])
    f_out = open(r'C:\Users\xichentop\Documents\573\lexrank_scores', 'w')
    for topic in all_docs:    
        eigen_vector, sent2idx = get_lexrank_scores(all_docs[topic], tf_idf_dict[topic]['tf_idf'], 0.1, 0.1, 0.1, False)
        for s in sent2idx:
            f_out.write('{} {}\n'.format(s, eigen_vector[sent2idx[s]]))
    
    f_out.close() 
    
def get_tf_idfs(docs):
    '''
    @param docs: a cluster of source documents in processed format
    return a dictionary with each word mapped to a tf_idf score 
    '''
    tf_idfs = {}
    doc_counts = {}
    word_counts = {}
    
    for doc in docs.values(): 
        flattened_list = [item for sublist in doc for item in sublist]
              
        for word in flattened_list:
            if word in word_counts:
                word_counts[word]+=1
            else:
                word_counts[word]=1
        
        vocab= set(flattened_list) 
        for term in vocab:
            if term in doc_counts:
                doc_counts[term]+=1
            else:
                doc_counts[term]=1
    for key in doc_counts:
        tf_idfs[key]= word_counts[key] * math.log(float(len(docs)) / doc_counts[key])   
    return tf_idfs    
                

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

def get_lexrank_scores(an_event, tf_idf_dict, threshold, error, damping_factor, basic):
    """
    @param an_event: a cluster of source documents in processed format
    @param tf_idf_dict: a word to tf_idf dict
    @param threshold: threshold for edges between nodes in range 0.1-0.3
    @param error: error for calculating the convergence of eigenvectors 0.1-0.2
    @param damping_factor: damping factor to facilitate convergence in range 0.1-0.2
    @param basic: True if use basic lexrank model; False to use continuous model
    @return: the lexrank scores for each indexed sentence and a lookup table for sentences       
    """
    all_sentences = []
    for doc in an_event.values():
#         all_sentences += doc
        #try unempty sentences only
        all_sentences+=[s for s in doc if s]
            
    cos_matrix, sent2idx, degree = get_cosine_sim_matrix(all_sentences, tf_idf_dict, threshold, basic)
    if basic:
        for i in degree:
            cos_matrix[i] /= degree[i]
    else:
        row_sums = cos_matrix.sum(axis=1)
#         print(row_sums)
        for i in range(len(row_sums)):
            if row_sums[i]!=0:
                cos_matrix[i] /= row_sums[i] 
        
    
    cos_matrix = get_transition_kernel(cos_matrix, damping_factor) 
    eigen_vector = power_method(cos_matrix, error)
    #test 
#     idx = 0 
#     for x in np.nditer(eigen_vector):
#         print('sent '+str(idx), ' score: '+str(x))
#         idx+=1
    #end test
    return eigen_vector, sent2idx


def get_cosine_sim_matrix(all_sentences, tf_idf_dict, threshold, basic):
    #remove stop words and lower_case words
    """
    @param all_sentences: a list of sentences
    @param tf_idf_dict: a lookup table for tf_idf
    @param threshold: the threshold for inclusion in cosine matrix
    @param basic: true to use the basic mode, false to use the continuous mode 
    @return a n x n cosine matrix (n=length of all_sentences)   
    """
    n = len(all_sentences)
    cos_matrix = np.zeros(shape=(n, n), dtype=float)
    degree = {}
    
    sent2idx = {}
    idx2sent = []
    idx = 0
    for s1, s2 in itertools.combinations_with_replacement(all_sentences, 2):
        s1_str = ' '.join(s1)
        s2_str = ' '.join(s2)
        if s1_str not in sent2idx:
            sent2idx[s1_str] = idx
            idx2sent.append(s1_str)
            idx+=1
            
        if s2_str not in sent2idx:
            sent2idx[s2_str] = idx
            idx2sent.append(s2_str)
            idx+=1
        # should make keys lower case previously 
        s1_tf_idf = np.array([tf_idf_dict.get(w1, 0) for w1 in s1])
        s2_tf_idf = np.array([tf_idf_dict.get(w2, 0) for w2 in s2])
        
        # test
#         print(s1_tf_idf)
#         print(s2_tf_idf)
        #test if the vectors are composed of all zeros
        if np.any(s1_tf_idf) and np.any(s2_tf_idf):
            #padding
            if s1_tf_idf.size > s2_tf_idf.size:
                s2_tf_idf = np.append(s2_tf_idf, [0.0]*(s1_tf_idf.size-s2_tf_idf.size))
                
            elif s1_tf_idf.size < s2_tf_idf.size:
                s1_tf_idf = np.append(s1_tf_idf, [0.0]*(s2_tf_idf.size-s1_tf_idf.size))
            
    #         print(s1_tf_idf, s2_tf_idf)
            
            cos_sim = 1-cosine(s1_tf_idf, s2_tf_idf)
        else: 
            cos_sim = 0.0
        #basic lexrank > threshold cos_sim = 1.0; else cos_sim = 0
        if basic:
            if cos_sim>threshold:
                s1_idx = sent2idx[s1_str]
                cos_matrix[s1_idx, sent2idx[s2_str]] = 1.0
                if s1_idx not in degree:
                    degree[s1_idx] = 0
                degree[s1_idx]+=1
        else:
            s1_idx = sent2idx[s1_str]
            cos_matrix[s1_idx, sent2idx[s2_str]] = cos_sim
#             if s1_idx not in degree:
#                 degree[s1_idx] = 0
#             degree[s1_idx]+=1
        
#         else:
#             cos_matrix[sent2idx[s1], sent2idx[s2]] = 0
    
#     print('cluster', cos_matrix)
    return cos_matrix, sent2idx, degree

def get_transition_kernel(cos_matrix, damping_factor):
    '''
    return the transition kernel
    '''
    n = len(cos_matrix)
    square_matrix = np.zeros(shape=(n, n))
    square_matrix.fill(1.0/n)
    transition_kernel = damping_factor*square_matrix + (1-damping_factor)*cos_matrix
    return transition_kernel 

test_lexrank(r'C:\Users\xichentop\workspace\573\project\573\src\data\training.processed.json')

