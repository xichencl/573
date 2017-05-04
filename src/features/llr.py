import math
import nltk

stops = set(nltk.corpus.stopwords.words('english'))

def get_back_counts(all_docs):
    background_counts = {}
    for event in all_docs:
        event_docs = all_docs[event]
        for event_doc in event_docs:
            a_doc = event_docs[event_doc]
            words = [item for sublist in a_doc for item in sublist]
            for word in words:
                if word in background_counts:
                    background_counts[word] += 1
                else:
                    background_counts[word] = 1
    return background_counts


def get_cluster_counts(cluster):
    cluster_counts = {}
    for doc in cluster:
        a_doc = cluster[doc]
        words = [item for sublist in a_doc for item in sublist]
        for word in words:
            if word in cluster_counts.keys():
                cluster_counts[word] += 1
            else:
                cluster_counts[word] = 1
    return cluster_counts


def get_weight_sum(sentence, background_c, cluster_c):
    sent_sum = 0
    words = sentence
    for word in words:
        if word in cluster_c.keys() and word in background_c.keys():
            k_one = cluster_c[word]
            k_two = background_c[word]
            n_one = len(cluster_c.keys())
            n_two = len(background_c.keys())
            p = float((k_one + k_two))/(n_one + n_two)
            if p > 0 and (1 - p) > 0:
                l_one = k_one * math.log(p) + (n_one - k_one) * math.log(1 - p)
                l_two = k_two * math.log(p) + (n_two - k_two) * math.log(1 - p)
                final_log = -l_one - l_two
            else:
                final_log = 0
            if final_log > 10:
                sent_sum += 1
    return [sent_sum, float(sent_sum)/len(words)]
