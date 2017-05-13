import re
from features import tf_idf
from features import llr
from features import ling_features
from features import kl
from features import kl_bigrams
from features import position
from features import lexrank
from features import q_kl
import Query
import random
import feature_select
import eval
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import nltk
import pickle
import json


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

    def vectorize(self, sentence, idx, doc_len, document, tf_idfs, back_counts, cluster_counts, an_event, back_list, vocab, back_list2, vocab2,
                  first_p, all_p, lexrank, q_lex, query):
        vec = []
        string = ' '.join(sentence)
        key = (string, document)
        if key in self.vecs:
            vec = self.vecs[key]
        else:
            vec.extend(tf_idf.get_tf_idf_average(sentence, tf_idfs))
            vec.extend(llr.get_weight_sum(sentence, back_counts, cluster_counts))
            vec.append(len(sentence))
            vec = ling_features.add_feats(an_event, sentence, vec)
            vec.extend(kl.get_kl(sentence, back_list, vocab))
            vec.extend(kl_bigrams.get_kl(sentence, back_list2, vocab2))
            vec.extend(position.score_sent(sentence, first_p, all_p))
            vec.append(int(idx < 1))
            vec.append(idx)
            vec.append(lexrank)
            vec.append(q_lex)
            vec.extend(q_kl.get_kl(sentence, query.split()))
            vec = np.array(vec)
            self.vecs[key] = vec
        return vec[:-2]

    # place any code that needs access to the gold standard summaries here
    def train(self, docs, gold):

        back_counts = llr.get_back_counts(docs)
        self.background_counts = back_counts

        event_ind = 1
        # process sentences in each document of each cluster
        x = []
        y = []
        for event in docs:
            unseen = False
            an_event = docs[event]
            if event not in self.cluster_info:
                unseen = True
                self.cluster_info[event] = {}
                tf_idfs = tf_idf.get_tf_idfs(an_event)
                self.cluster_info[event]["tf_idf"] = tf_idfs
                sum_words = []
                for document in gold[event]:
                    if document[2] == '7' or document[6] == 'A':
                        a_sum = gold[event][document]
                        for sent in a_sum:
                            for word in sent:
                                if isinstance(word, list):
                                    for real_word in word:
                                        sum_words.append(real_word)
                                else:
                                    sum_words.append(word)
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

                lex_results = lexrank.get_lexrank_scores(an_event, self.cluster_info[event]["tf_idf"], 0.1, 0.1,
                                                             0.1, False)
                self.cluster_info[event]["eigen"] = lex_results[0]
                self.cluster_info[event]["sent2idx"] = lex_results[1]
                if "Group" in event:
                    sums = {}
                    idx = 0
                    for sum in gold[event]["100"]:
                        sums[str(idx)] = sum
                        idx += 1
                    g_eigen, g_idx = lexrank.get_lexrank_scores(sums, self.cluster_info[event]["tf_idf"], 0.1, 0.1,
                                                                0.1, False)
                    self.cluster_info[event]["g_eigen"] = g_eigen
                    self.cluster_info[event]["g_idx"] = g_idx

                else:
                    g_eigen, g_idx = lexrank.get_lexrank_scores(gold[event], self.cluster_info[event]["tf_idf"], 0.1, 0.1,
                                                                0.1, False)
                    self.cluster_info[event]["g_eigen"] = g_eigen
                    self.cluster_info[event]["g_idx"] = g_idx

                topics = json.load(open('../src/data/training.topic_dict.reverse.json', 'r'))
                query = topics[event]
                rel_scores, sentences = Query.get_rel_scores(an_event, query.split())
                q_lex_result = Query.get_lexrank_scores(an_event, self.cluster_info[event]["tf_idf"], rel_scores, 0.2,
                                                             0.1,
                                                             0.95,
                                                             False)
                g_rel_scores, g_sentences = Query.get_rel_scores(gold[event], query.split())
                g_q_lex_result = Query.get_lexrank_scores(gold[event], self.cluster_info[event]["tf_idf"],
                                                                 g_rel_scores, 0.2, 0.1,
                                                                 0.95,
                                                                 False)
                self.cluster_info[event]['q_lex'] = q_lex_result[0]
                self.cluster_info[event]['q_sent2idx'] = q_lex_result[1]
                self.cluster_info[event]['g_q_lex'] = g_q_lex_result[0]
                self.cluster_info[event]['g_q_sent2idx'] = g_q_lex_result[1]

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
            eigen = self.cluster_info[event]["eigen"]
            sent2idx = self.cluster_info[event]["sent2idx"]
            g_eigen = self.cluster_info[event]["g_eigen"]
            g_idx = self.cluster_info[event]["g_idx"]

            q_lex = self.cluster_info[event]['q_lex']
            q_sent2idx = self.cluster_info[event]['q_sent2idx']
            g_q_lex = self.cluster_info[event]['g_q_lex']
            g_q_sent2idx = self.cluster_info[event]['g_q_sent2idx']

            topics = json.load(open('../src/data/training.topic_dict.reverse.json', 'r'))
            query = topics[event]

            for document in an_event:
                a_doc = an_event[document]
                sent_idx = 0
                for sentence in a_doc:
                    # construct a vector for each sentence in the document
                    if 1 < len(sentence):

                        vec = self.vectorize(sentence, sent_idx, len(a_doc), document, self.cluster_info[event]["tf_idf"], back_counts, cluster_counts,
                                             an_event, back_list, vocab, back_list2, vocab2, first_p, all_p, eigen[sent2idx[" ".join(sentence)]], q_lex[q_sent2idx[' '.join(sentence)]], query)
                        sent_idx += 1
                        # Add additional features here
                        x.append(vec)
                        y.append(eval.get_rouge(sentence, sum_bigs, sum_tri, sum_quad, sum_words_fd))

            gold_sums = gold[event]
            for document in gold_sums:
                if len(document) < 6 or document[6] == 'A':
                    a_sum = gold_sums[document]
                    sents = []
                    if "Group" in event and document == "100":
                        for one_sum in gold_sums[document]:
                            for a_sent in one_sum:
                                sents.append(a_sent)
                    elif 'Group' not in event:
                        sents = gold_sums[document]

                    # construct a vector for each sentence in the summary
                    sent_idx = 0
                    for sentence in sents:
                        if len(sentence) > 1:
                            vec = self.vectorize(sentence, sent_idx, len(a_sum),  document, self.cluster_info[event]["tf_idf"], back_counts, cluster_counts,
                                                 an_event, back_list, vocab, back_list2, vocab2, first_p, all_p, g_eigen[g_idx[" ".join(sentence)]],  g_q_lex[g_q_sent2idx[' '.join(sentence)]], query)
                            sent_idx += 1
                            # Add additional features here
                            x.append(vec)
                            y.append(eval.get_rouge(sentence, sum_bigs, sum_tri, sum_quad, sum_words_fd))
            #if unseen:
            vec_file = open('vecs.p', 'wb')
            pickle.dump(self.vecs, vec_file)
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        y = np.array(y) / max(y)

        self.model = MLPRegressor(hidden_layer_sizes=[50,])
        self.model.fit(x, y)
        feature_select.get_feats(x, y)

    def test(self, docs, preproc, name, query=None):
        unseen = False
        if name not in self.cluster_info:
            unseen = True
            self.cluster_info[name] = {}
            tf_idfs = tf_idf.get_tf_idfs(preproc)
            self.cluster_info[name]["tf_idf"] = tf_idfs

            pos_results = position.get_positions(preproc)
            self.cluster_info[name]["first_p"] = pos_results[0]
            self.cluster_info[name]["all_p"] = pos_results[1]

            kl_result = kl.get_freq_list(preproc)
            self.cluster_info[name]["back_list"] = kl_result[0]
            self.cluster_info[name]["vocab"] = kl_result[1]

            kl_result2 = kl_bigrams.get_freq_list(preproc)
            self.cluster_info[name]["back_list2"] = kl_result2[0]
            self.cluster_info[name]["vocab2"] = kl_result2[1]

            self.cluster_info[name]["cluster_counts"] = llr.get_cluster_counts(preproc)

            lex_results = lexrank.get_lexrank_scores(preproc, self.cluster_info[name]["tf_idf"], 0.1, 0.1,
                                                     0.1, False)
            self.cluster_info[name]["eigen"] = lex_results[0]
            self.cluster_info[name]["sent2idx"] = lex_results[1]

            topics = json.load(open('../src/data/training.topic_dict.reverse.json', 'r'))
            query = topics[name]
            rel_scores, sentences = Query.get_rel_scores(preproc, query.split())
            q_lex_result = Query.get_lexrank_scores(preproc, self.cluster_info[name]["tf_idf"], rel_scores, 0.2,
                                                    0.1,
                                                    0.95,
                                                    False)

            self.cluster_info[name]['q_lex'] = q_lex_result[0]
            self.cluster_info[name]['q_sent2idx'] = q_lex_result[1]

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
        eigen = self.cluster_info[name]["eigen"]
        sent2idx = self.cluster_info[name]["sent2idx"]

        q_lex = self.cluster_info[name]['q_lex']
        q_sent2idx = self.cluster_info[name]['q_sent2idx']

        topics = json.load(open('../src/data/training.topic_dict.reverse.json', 'r'))
        query = topics[name]

        for document in docs:
            doc_sents = []
            a_doc = docs[document]
            proc_doc = preproc[document]
            index = 0

            # construct a vector for each sentence in the document
            sent_idx = 0
            for i in range(len(a_doc)):
                sentence = a_doc[i]
                proc_sent = proc_doc[i]
                index += 1
                sentence = re.sub('\n', ' ', sentence)
                if 7 < len(sentence.split()) < 22:
                    vec = self.vectorize(proc_sent, sent_idx, len(a_doc),  document, self.cluster_info[name]["tf_idf"], self.background_counts, cluster_counts,
                                         docs, back_list, vocab, back_list2, vocab2, first_p, all_p, eigen[sent2idx[' '.join(proc_sent)]], q_lex[q_sent2idx[" ".join(proc_sent)]], query)
                    sent_idx += 1
                    vec = vec.reshape(1, -1)
                    vec = self.scaler.transform(vec)

                    # Add additional features here
                    doc_sents.append((sentence, float(self.model.predict(vec))))
                    all_sents.append((sentence, float(self.model.predict(vec))))
            sents[document] = doc_sents
        if unseen:
            vec_file = open('vecs.p', 'wb')
            pickle.dump(self.vecs, vec_file)
        return sents
