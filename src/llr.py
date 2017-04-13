import math
import nltk


def get_back_counts(all_docs):
    background_counts = {}
    for event in all_docs.keys():
        event_docs = all_docs[event]
        for event_doc in event_docs.keys():
            a_doc = event_docs[event_doc]
            words = nltk.word_tokenize(' '.join(a_doc))
            for word in words:
                if word in background_counts.keys():
                    background_counts[word] += 1
                else:
                    background_counts[word] = 1
    return background_counts


def get_cluster_counts(cluster):
    cluster_counts = {}
    for doc in cluster.keys():
        a_doc = cluster[doc]
        words = nltk.word_tokenize(' '.join(a_doc))
        for word in words:
            if word in cluster_counts.keys():
                cluster_counts[word] += 1
            else:
                cluster_counts[word] = 1
    return cluster_counts


def get_weight_sum(sentence, background_c, cluster_c):
    sent_sum = 0
    words = nltk.word_tokenize(sentence)
    for word in words:
        if word in cluster_c.keys() and word in background_c.keys():
            k_one = cluster_c[word]
            k_two = background_c[word]
            n_one = len(cluster_c.keys())
            n_two = len(background_c.keys())
            p = float((k_one + k_two))/(n_one + n_two)
            l_one = p ** (k_one * (1 - p)**(n_one - k_one))
            l_two = p ** (k_two * (1 - p)**(n_two - k_two))
            final_log = -math.log(l_one) - math.log(l_two)
            if final_log > 10:
                sent_sum += float(1)/len(words)
    return sent_sum