import re
from features import tf_idf
from features import llr
from features import ling_features
from features import kl
from features import kl_bigrams
from features import position
import feature_select
import eval
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import math
import nltk


class ContentSelector:
    def __init__(self):
        self.model = None
        self.background_counts = None
        self.scaler = StandardScaler()

    def vectorize(self, sentence, cluster_info, event, an_event, back_counts, cluster_counts, back_list, vocab, first, all):
        vec = []
        vec.extend(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
        vec.extend(llr.get_weight_sum(sentence, back_counts, cluster_counts))
        vec.append(len(sentence.split()))
        vec = ling_features.add_feats(an_event, sentence, vec)
        vec.extend(kl.get_kl(sentence, back_list, vocab))
        vec.extend(position.score_sent(sentence, first, all))
        vec = np.array(vec)
        return vec

    # place any code that needs access to the gold standard summaries here
    def train(self, docs, gold):

        # dictionary containing feature information for each document cluster
        cluster_info = {}
        for event in docs.keys():
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
        for event in cluster_info.keys():
            '''
            all_sums = ''
            for document in gold[event].keys():
                if len(document) < 6 or document[6] == 'A':
                    a_sum = gold[event][document]
                    if isinstance(a_sum, list):
                        a_sum = ' '.join(a_sum)
                    a_sum = re.sub('\n', ' ', a_sum)
                    all_sums += a_sum + ' '
            sum_words = nltk.word_tokenize(all_sums)
            sum_bigs = list(nltk.ngrams(sum_words, 2))'''


            print('Processing Cluster ' + str(event_ind) + '/' + str(len(cluster_info.keys())))
            event_ind += 1
            an_event = docs[event]
            first, all = position.get_positions(an_event)
            back_list, vocab = kl.get_freq_list(an_event)
            back_list2, vocab2 = kl_bigrams.get_freq_list(an_event)
            cluster_counts = llr.get_cluster_counts(an_event)
            for document in an_event.keys():
                a_doc = an_event[document]
                for sentence in a_doc:
                    sentence = re.sub('\n', ' ', sentence)

                    # construct a vector for each sentence in the document
                    if 1 < len(sentence.split()):
                        vec = []
                        vec.extend(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                        vec.extend(llr.get_weight_sum(sentence, back_counts, cluster_counts))
                        vec.append(len(sentence.split()))
                        vec = ling_features.add_feats(an_event, sentence, vec)
                        vec.extend(kl.get_kl(sentence, back_list, vocab))
                        vec.extend(kl_bigrams.get_kl(sentence, back_list2, vocab2))
                        vec.extend(position.score_sent(sentence, first, all))
                        vec = np.array(vec)
                        # Add additional features here
                        x.append(vec)
                        y.append(0)
            gold_sums = gold[event]
            for document in gold_sums.keys():
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
                            vec.append(len(sentence.split()))
                            vec = ling_features.add_feats(an_event, sentence, vec)
                            vec.extend(kl.get_kl(sentence, back_list, vocab))
                            vec.extend(kl_bigrams.get_kl(sentence, back_list2, vocab2))
                            vec.extend(position.score_sent(sentence, first, all))
                            vec = np.array(vec)
                            # Add additional features here
                            x.append(vec)
                            y.append(1)
        #self.scaler.fit(x)
        #x = self.scaler.transform(x)
        y = np.array(y) / max(y)
        #parameters = {'alpha': 10.0 ** -np.arange(1, 7), 'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #              'solver': ['lbfgs', 'sgd', 'adam']}

        #self.model = GridSearchCV(MLPRegressor(), parameters)
        #self.model = MLPRegressor()
        self.model = LinearRegression()
        self.model.fit(x, y)
        #print(self.model)
        feature_select.get_feats(x, y)

    def test(self, docs, query=None):
        info = {}
        info['tf_idf'] = tf_idf.get_tf_idfs(docs)
        sents = []
        cluster_counts = llr.get_cluster_counts(docs)
        back_list, vocab = kl.get_freq_list(docs)
        back_list2, vocab2 = kl_bigrams.get_freq_list(docs)
        first, all = position.get_positions(docs)
        for document in docs.keys():
            a_doc = docs[document]
            index = 0
            #if len(a_doc) > 1:
            #    sents.append((a_doc[1], 1))
            #else:
            #    sents.append((a_doc[0], 1))

            # construct a vector for each sentence in the document
            for sentence in a_doc:
                index += 1
                sentence = re.sub('\n', ' ', sentence)
                if 7 < len(sentence.split()) < 22:
                    vec = []
                    vec.extend(tf_idf.get_tf_idf_average(sentence, info["tf_idf"]))
                    vec.extend(llr.get_weight_sum(sentence, self.background_counts, cluster_counts))
                    vec.append(len(sentence.split()))
                    vec = ling_features.add_feats(docs, sentence, vec)
                    vec.extend(kl.get_kl(sentence, back_list, vocab))
                    vec.extend(kl_bigrams.get_kl(sentence, back_list2, vocab2))
                    vec.extend(position.score_sent(sentence, first, all))
                    vec = np.array(vec).reshape(1, -1)
                    # vec = self.scaler.transform(vec)

                    # Add additional features here
                    sents.append((sentence, self.model.predict(vec)))
        return sorted(sents, key=lambda x: x[1], reverse=True)
