"""
Implements a query-focused lexrank summarizer.
Returns a dictionary mapping the topic queried to a list of tuples, 
each containing a space-delimited processed sentence and a score for the sentence
in descending order. 
main method: get_lexrank_query
"""

import nltk
import numpy as np
from scipy.spatial.distance import cosine
import itertools
import re
import math
import json
import sys
import operator

stops = set(nltk.corpus.stopwords.words('english'))

def get_lexrank_query(file, topic_dict, topic, query):
    '''
    @param file: the json file to be loaded
    @topic_dict: a dict mapping topics to topic_ids 
    @param topic: the topic to be queried
    @param query: query string
    '''
    #find topic_id
    topic= topic.lower()
    if topic in topic_dict:
        topic_id = topic_dict[topic]
    else:
        sys.stderr.print("There's no such topic.")
        exit()
    
    all_docs = json.load(open(file, 'r'))
    #process query
    query = [word for word in query.lower().split() if word not in stops and re.match('\w', word)]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tags = nltk.pos_tag(query)
    idx = 0
    for w, t in tags:
        if t.startswith('V'):
            query[idx] = lemmatizer.lemmatize(w, 'v')
        elif t.startswith('N'):
            query[idx] = lemmatizer.lemmatize(w, 'n')
        idx+=1
    
    
    rel_scores, sentences = get_rel_scores(all_docs[topic_id], query)
    tf_idf_dict = get_tf_idfs(all_docs[topic_id])
    lexrank_scores, sent2idx = get_lexrank_scores(all_docs[topic_id], tf_idf_dict, rel_scores, 0.2, 0.1, 0.95, False)
    rel_sents = {}
    rel_sents[topic] = []
    for sent in sent2idx:
        idx = sent2idx[sent]
        rel_sents[topic].append((sent, lexrank_scores[idx]))
    sorted_lexrank = sorted(rel_sents[topic].items(), key=operator.itemgetter(1), reverse=True)
    return sorted_lexrank

    
    
    
#     f_out = open(r'C:\Users\xichentop\Documents\573\lexrank_scores', 'w')
#     for topic in all_docs:    
#         eigen_vector, sent2idx = get_lexrank_scores(all_docs[topic], tf_idf_dict[topic]['tf_idf'], 0.1, 0.1, 0.1, False)
#         for s in sent2idx:
#             f_out.write('{} {}\n'.format(s, eigen_vector[sent2idx[s]]))
#     
#     f_out.close() 
    
def get_rel_scores(docs, query):
    '''
    @param docs: a cluster of source documents in processed format
    @param query: a processed list of query terms
    @return a dictionary with each sentence mapped to a relevance score 
    and a list of non-empty sentences  
    '''
    
    idfs= {}
    sent_counts = {}
    sentences = []
    for doc in docs.values():
        sentences += doc
    
    rel_scores = np.zeros(len(sentences))
    
    tf_sent = {}
#     sent_idx = []
    idx = 0
    for s in sentences: 
        for term in set(query):
            if term in s:
#                 sent_idx.append(idx)
                if term in sent_counts:
                    sent_counts[term]+= 1
                else:
                    sent_counts[term]= 1
                if term not in tf_sent:
                    tf_sent[term] = []
                tf_sent[term].append((idx, s.count(term)))
        idx+=1
    for key in sent_counts:
        idfs[key]= math.log(float(len(sentences)+1) / (0.5+ sent_counts[key]))

    tf_query ={}
    for word in query:
        if word in tf_query:
            tf_query[word]+=1
        else:
            tf_query[word]=1
    
    for word in tf_sent:
        for idx, count in tf_sent[word]:
            rel_scores[idx] += math.log(tf_query[word]+1)*math.log(count+1)*idfs[word]         
    
    return rel_scores, sentences   
                
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
    idx = 0
    while delta >= error:
        v2 = np.dot(cos_matrix.T, v1)
        print(str(idx)+'iteration, vector: ', v2)
        
        delta = np.linalg.norm(np.subtract(v2,v1))
        print(str(idx)+'iteration, delta: ', delta)
        idx+=1
        v1 = v2
#     print(v1)
    return v1

def get_lexrank_scores(docs, tf_idf_dict, rel_scores, threshold, error, damping_factor, basic):
    """
    @param docs: a cluster of source documents in processed format
#     @param tf_idf_dict: a word to tf_idf dict
    @param threshold: threshold for edges between nodes, best range 0.14-0.2
    @param error: error for calculating the convergence of eigenvectors 0.1-0.2
    @param damping_factor: damping factor to facilitate convergence. best range: 0.8-0.95
    @param basic: True if use basic lexrank model; False to use continuous model
    @return: the lexrank scores for each indexed sentence and a lookup table for sentences       
    """
    
#     tf_idf_dict = get_tf_idfs(docs)
    all_sentences = []
    for doc in docs.values():
        all_sentences += doc
        #try unempty sentences only
#         all_sentences+=[s for s in doc if s]
            
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
        
    
    cos_matrix = get_transition_kernel(cos_matrix, rel_scores, damping_factor) 
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
            if cos_sim>threshold:
                s1_idx = sent2idx[s1_str]
                cos_matrix[s1_idx, sent2idx[s2_str]] = cos_sim
#             if s1_idx not in degree:
#                 degree[s1_idx] = 0
#             degree[s1_idx]+=1
        
#         else:
#             cos_matrix[sent2idx[s1], sent2idx[s2]] = 0
    
#     print('cluster', cos_matrix)
    return cos_matrix, sent2idx, degree

def get_transition_kernel(cos_matrix, rel_scores, damping_factor):
    '''
    return the transition kernel
    '''
    n = len(cos_matrix)
    square_matrix = np.zeros(shape=(n, n))
    rel_scores /= rel_scores.sum()
    print('normalized rel-scores: ', rel_scores)
    for i in range(n):
        square_matrix[i] = rel_scores
    transition_kernel = damping_factor*square_matrix + (1-damping_factor)*cos_matrix
    return transition_kernel 

def test_lexrank_query():
    json_file = r'C:\Users\xichentop\workspace\573\project\573\src\data\training.processed.json'
    dict = json.load(open(json_file, 'r'))
    rel_scores, sentences = get_rel_scores(dict['D0917C'], ['who', 'new', 'pope'])
    print('rel scores: ', rel_scores)
    tf_idf_dict = get_tf_idfs(dict['D0917C'])
    lexrank_scores, sent2idx = get_lexrank_scores(dict['D0917C'], tf_idf_dict, rel_scores, 0.1, 0.1, 0.5, False)
    rel_sents = []
    for sent in sent2idx:
        idx = sent2idx[sent]
        rel_sents.append((sent, lexrank_scores[idx]))
    sorted_lexrank = sorted(rel_sents, key=operator.itemgetter(1), reverse=True)
    for sent, score in sorted_lexrank:
        print(sent, score)
#     idx = 0
#     for score in rel_scores:
#         if score != 0.0:
#             print(sentences[idx], score) 
#         idx+=1
test_lexrank_query()
