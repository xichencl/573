import re
from features import tf_idf
from features import llr
from features import ling_features
from features import kl
from features import kl_bigrams
from features import position
import random
import feature_select
import eval
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import nltk

class ContentSelector:
    def __init__(self):
        self.model = None
        self.background_counts = None
        self.scaler = StandardScaler()

    # place any code that needs access to the gold standard summaries here
    def train(self, docs, gold):

        # dictionary containing feature information for each document cluster
        cluster_info = {}
        for event in docs:
            an_event = docs[event]
            tf_idfs = tf_idf.get_tf_idfs(an_event)
            cluster_info[event] = {}
            cluster_info[event]["tf_idf"] = tf_idfs
        back_counts = llr.get_back_counts(docs)
        self.background_counts = back_counts

        event_ind = 1
        # process sentences in each document of each cluster
        x = []
        y = []
        for event in cluster_info:
            all_sums = ''
            for document in gold[event]:
                if len(document) < 6 or document[6] == 'A':
                    a_sum = gold[event][document]
                    if isinstance(a_sum, list):
                        a_sum = ' '.join(a_sum)
                    a_sum = re.sub('\n', ' ', a_sum)
                    all_sums += a_sum + ' '
            sum_words = nltk.word_tokenize(all_sums)
            sum_bigs = list(nltk.ngrams(sum_words, 2))
            sum_tri = list(nltk.ngrams(sum_words, 3))
            sum_quad = list(nltk.ngrams(sum_words, 4))

            print('Processing Cluster ' + str(event_ind) + '/' + str(len(cluster_info.keys())))
            event_ind += 1
            an_event = docs[event]
            first, all = position.get_positions(an_event)
            back_list, vocab = kl.get_freq_list(an_event)
            back_list2, vocab2 = kl_bigrams.get_freq_list(an_event)
            cluster_counts = llr.get_cluster_counts(an_event)
            for document in an_event:
                a_doc = an_event[document]
                for sentence in a_doc:
                    sentence = re.sub('\n', ' ', sentence)

                    # construct a vector for each sentence in the document
                    if 1 < len(sentence.split()):
                        vec = []
                        vec.extend(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                        vec.extend(llr.get_weight_sum(sentence, back_counts, cluster_counts))
                        #vec.append(len(sentence.split()))
                        vec = ling_features.add_feats(an_event, sentence, vec)
                        vec.extend(kl.get_kl(sentence, back_list, vocab))
                        vec.extend(kl_bigrams.get_kl(sentence, back_list2, vocab2))
                        vec.extend(position.score_sent(sentence, first, all))
                        vec = np.array(vec)
                        # Add additional features here
                        x.append(vec)
                        y.append(eval.get_rouge(sentence, sum_bigs, sum_tri, sum_quad, list(sum_words)))
            gold_sums = gold[event]
            for document in gold_sums:
                if len(document) < 6 or document[6] == 'A':
                    a_sum = gold_sums[document]
                    if isinstance(a_sum, list):
                        a_sum = ' '.join(a_sum)
                    a_sum = re.sub('\n', ' ', a_sum)

                    # construct a vector for each sentence in the summary
                    sents = nltk.sent_tokenize(a_sum)
                    for sentence in sents:
                        if len(sentence) > 1:
                            vec = []
                            vec.extend(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                            vec.extend(llr.get_weight_sum(sentence, back_counts, cluster_counts))
                            #vec.append(len(sentence.split()))
                            vec = ling_features.add_feats(an_event, sentence, vec)
                            vec.extend(kl.get_kl(sentence, back_list, vocab))
                            vec.extend(kl_bigrams.get_kl(sentence, back_list2, vocab2))
                            vec.extend(position.score_sent(sentence, first, all))
                            vec = np.array(vec)
                            # Add additional features here
                            x.append(vec)
                            y.append(eval.get_rouge(sentence, sum_bigs, sum_tri, sum_quad, list(sum_words)))
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        y = np.array(y) / max(y)

        self.model = MLPRegressor()
        self.model.fit(x, y)
        feature_select.get_feats(x, y)

    def test(self, docs, query=None):
        info = {}
        info['tf_idf'] = tf_idf.get_tf_idfs(docs)
        sents = {}
        cluster_counts = llr.get_cluster_counts(docs)
        back_list, vocab = kl.get_freq_list(docs)
        back_list2, vocab2 = kl_bigrams.get_freq_list(docs)
        first, all = position.get_positions(docs)
        for document in docs:
            doc_sents = []
            a_doc = docs[document]
            index = 0
            if len(a_doc) > 1:
                doc_sents.append((a_doc[1], 1))
            else:
                doc_sents.append((a_doc[0], 1))

            # construct a vector for each sentence in the document
            for sentence in a_doc:
                index += 1
                sentence = re.sub('\n', ' ', sentence)
                if 7 < len(sentence.split()) < 22:
                    vec = []
                    vec.extend(tf_idf.get_tf_idf_average(sentence, info["tf_idf"]))
                    vec.extend(llr.get_weight_sum(sentence, self.background_counts, cluster_counts))
                    #vec.append(len(sentence.split()))
                    vec = ling_features.add_feats(docs, sentence, vec)
                    vec.extend(kl.get_kl(sentence, back_list, vocab))
                    vec.extend(kl_bigrams.get_kl(sentence, back_list2, vocab2))
                    vec.extend(position.score_sent(sentence, first, all))
                    vec = np.array(vec).reshape(1, -1)
                    vec = self.scaler.transform(vec)

                    # Add additional features here
                    doc_sents.append((sentence, float(self.model.predict(vec))))
            sents[document] = doc_sents
        return sents
