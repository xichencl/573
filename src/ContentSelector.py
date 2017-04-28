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
import pickle


class ContentSelector:
    def __init__(self):
        self.model = None
        self.background_counts = None
        self.scaler = StandardScaler()
        self.vecs = {}
        try:
            p_file = open('vecs.p', 'rb')
            self.vecs = pickle.load(p_file)
        except FileNotFoundError:
            pass
        self.cluster_info = {}
        try:
            c_file = open('cluster_info.p', 'rb')
            self.cluster_info = pickle.load(c_file)
        except FileNotFoundError:
            pass

    def vectorize(self, sentence, document, tf_idfs, back_counts, cluster_counts, an_event, back_list, vocab, back_list2, vocab2,
                  first_p, all_p):
        vec = []
        key = (sentence, document)
        if key in self.vecs:
            vec = self.vecs[key]
        else:
            vec.extend(tf_idf.get_tf_idf_average(sentence, tf_idfs))
            vec.extend(llr.get_weight_sum(sentence, back_counts, cluster_counts))
            #vec.append(len(sentence.split()))
            vec = ling_features.add_feats(an_event, sentence, vec)
            vec.extend(kl.get_kl(sentence, back_list, vocab))
            vec.extend(kl_bigrams.get_kl(sentence, back_list2, vocab2))
            vec.extend(position.score_sent(sentence, first_p, all_p))
            vec = np.array(vec)
            self.vecs[key] = vec
        return vec

    # place any code that needs access to the gold standard summaries here
    def train(self, docs, gold):

        back_counts = llr.get_back_counts(docs)
        self.background_counts = back_counts

        event_ind = 1
        # process sentences in each document of each cluster
        x = []
        y = []
        for event in docs:
            an_event = docs[event]
            if event not in self.cluster_info:
                self.cluster_info[event] = {}
                tf_idfs = tf_idf.get_tf_idfs(an_event)
                self.cluster_info[event]["tf_idf"] = tf_idfs
                all_sums = ''
                for document in gold[event]:
                    if len(document) < 6 or document[6] == 'A':
                        a_sum = gold[event][document]
                        if isinstance(a_sum, list):
                            a_sum = ' '.join(a_sum)
                        a_sum = re.sub('\n', ' ', a_sum)
                        all_sums += a_sum + ' '
                sum_words = nltk.word_tokenize(all_sums)
                self.cluster_info[event]["sum_words_fd"] = nltk.FreqDist(sum_words)
                self.cluster_info[event]["sum_bigs"] = nltk.FreqDist(list(nltk.ngrams(sum_words, 2)))
                self.cluster_info[event]["sum_tri"] = nltk.FreqDist(list(nltk.ngrams(sum_words, 3)))
                self.cluster_info[event]["sum_quad"] = nltk.FreqDist(list(nltk.ngrams(sum_words, 4)))

                pos_results = position.get_positions(an_event)
                self.cluster_info[event]["first_p"] = pos_results[0]
                self.cluster_info[event]["all_p"] = pos_results[1]

                kl_result = kl.get_freq_list(an_event)
                self.cluster_info[event]["back_list"] = kl_result[0]
                self.cluster_info[event]["vocab"] = kl_result[1]

                kl_result2 = kl_bigrams.get_freq_list(an_event)
                self.cluster_info[event]["back_list2"] = kl_result2[0]
                self.cluster_info[event]["vocab2"] = kl_result2[1]

                self.cluster_info[event]["cluster_counts"] = llr.get_cluster_counts(an_event)

                cluster_file = open('cluster_info.p', 'wb')
                pickle.dump(self.cluster_info, cluster_file)
            sum_words_fd = self.cluster_info[event]["sum_words_fd"]
            sum_bigs = self.cluster_info[event]["sum_bigs"]
            sum_tri = self.cluster_info[event]["sum_tri"]
            sum_quad = self.cluster_info[event]["sum_quad"]

            print('Processing Cluster ' + str(event_ind) + '/' + str(len(docs)))
            event_ind += 1
            first_p = self.cluster_info[event]["first_p"]
            all_p = self.cluster_info[event]["all_p"]
            back_list = self.cluster_info[event]["back_list"]
            vocab = self.cluster_info[event]["vocab"]
            back_list2 = self.cluster_info[event]["back_list2"]
            vocab2 = self.cluster_info[event]["vocab2"]
            cluster_counts = self.cluster_info[event]["cluster_counts"]
            for document in an_event:
                a_doc = an_event[document]
                for sentence in a_doc:
                    sentence = re.sub('\n', ' ', sentence)

                    # construct a vector for each sentence in the document
                    if 1 < len(sentence.split()):
                        vec = self.vectorize(sentence, document, self.cluster_info[event]["tf_idf"], back_counts, cluster_counts,
                                             an_event, back_list, vocab, back_list2, vocab2, first_p, all_p)
                        # Add additional features here
                        x.append(vec)
                        y.append(eval.get_rouge(sentence, sum_bigs, sum_tri, sum_quad, sum_words_fd))
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
                            vec = self.vectorize(sentence, document, self.cluster_info[event]["tf_idf"], back_counts, cluster_counts,
                                                 an_event, back_list, vocab, back_list2, vocab2, first_p, all_p)
                            # Add additional features here
                            x.append(vec)
                            y.append(eval.get_rouge(sentence, sum_bigs, sum_tri, sum_quad, sum_words_fd))
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        y = np.array(y) / max(y)

        self.model = MLPRegressor()
        self.model.fit(x, y)
        feature_select.get_feats(x, y)

    def test(self, docs, name, query=None):
        if name not in self.cluster_info:
            self.cluster_info[name] = {}
            tf_idfs = tf_idf.get_tf_idfs(docs)
            self.cluster_info[name]["tf_idf"] = tf_idfs

            pos_results = position.get_positions(docs)
            self.cluster_info[name]["first_p"] = pos_results[0]
            self.cluster_info[name]["all_p"] = pos_results[1]

            kl_result = kl.get_freq_list(docs)
            self.cluster_info[name]["back_list"] = kl_result[0]
            self.cluster_info[name]["vocab"] = kl_result[1]

            kl_result2 = kl_bigrams.get_freq_list(docs)
            self.cluster_info[name]["back_list2"] = kl_result2[0]
            self.cluster_info[name]["vocab2"] = kl_result2[1]

            self.cluster_info[name]["cluster_counts"] = llr.get_cluster_counts(docs)
            cluster_file = open('cluster_info.p', 'wb')
            pickle.dump(self.cluster_info, cluster_file)
        sents = {}
        all_sents = []
        cluster_counts = self.cluster_info[name]["cluster_counts"]
        back_list = self.cluster_info[name]["back_list"]
        vocab = self.cluster_info[name]["vocab"]
        back_list2 = self.cluster_info[name]["back_list2"]
        vocab2 = self.cluster_info[name]["vocab2"]
        first_p = self.cluster_info[name]["first_p"]
        all_p = self.cluster_info[name]["all_p"]
        for document in docs:
            doc_sents = []
            a_doc = docs[document]
            index = 0
            if len(a_doc) > 1:
                doc_sents.append((a_doc[1], 1))
                all_sents.append((a_doc[1], 1))
            else:
                doc_sents.append((a_doc[0], 1))
                all_sents.append((a_doc[0], 1))
            # construct a vector for each sentence in the document
            for sentence in a_doc:
                index += 1
                sentence = re.sub('\n', ' ', sentence)
                if 7 < len(sentence.split()) < 22:
                    vec = self.vectorize(sentence, document, self.cluster_info[name]["tf_idf"], self.background_counts, cluster_counts,
                                         docs, back_list, vocab, back_list2, vocab2, first_p, all_p)
                    vec = vec.reshape(1, -1)
                    vec = self.scaler.transform(vec)

                    # Add additional features here
                    doc_sents.append((sentence, float(self.model.predict(vec))))
                    all_sents.append((sentence, float(self.model.predict(vec))))
            sents[document] = doc_sents
        vec_file = open('vecs.p', 'wb')
        pickle.dump(self.vecs, vec_file)
        return all_sents
