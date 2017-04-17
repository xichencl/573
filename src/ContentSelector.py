import re
import tf_idf
import llr
import ling_features
import feature_select
import simplify_sent
import kl
from sklearn.neural_network import MLPRegressor
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

        # process sentences in each document of each cluster
        x = []
        y = []
        gold_lengths = []
        for event in cluster_info.keys():
            an_event = docs[event]
            back_list, vocab = kl.get_freq_list(an_event)
            cluster_counts = llr.get_cluster_counts(an_event)
            index = 0
            for document in an_event.keys():
                a_doc = an_event[document]
                for sentence in a_doc:

                    index += 1
                    # construct a vector for each sentence in the document

                    if len(sentence.split()) > 1:
                        vec = []
                        vec.extend(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                        vec.extend(llr.get_weight_sum(sentence, back_counts, cluster_counts))
                        vec.append(len(sentence.split()))
                        vec = ling_features.add_feats(an_event, sentence, vec)
                        vec.extend(kl.get_kl(sentence, back_list, vocab))
                        vec = np.array(vec)

                        # Add additional features here
                        x.append(vec)
                        y.append(0)
            gold_sums = gold[event]
            for document in gold_sums.keys():
                if document[6] == 'A':
                    a_sum = gold_sums[document]
                    a_sum = re.sub('\n', ' ', a_sum)

                    # construct a vector for each sentence in the summary
                    for sentence in nltk.sent_tokenize(a_sum):
                        if len(sentence) > 1:
                            gold_lengths.append(len(sentence.split()))
                            vec = []
                            vec.extend(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                            vec.extend(llr.get_weight_sum(sentence, back_counts, cluster_counts))
                            vec.append(len(sentence.split()))
                            vec = ling_features.add_feats(an_event, sentence, vec)
                            vec.extend(kl.get_kl(sentence, back_list, vocab))
                            vec = np.array(vec)

                            # Add additional features here
                            x.append(vec)
                            y.append(1)
        self.scaler.fit(x)
        x = self.scaler.transform(x)

        #parameters = {'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes': [(100,), (100, 100), (50, 50), (100, 50), (50, 100)],
        #             'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam']}

        #self.model = GridSearchCV(MLPRegressor(), parameters)
        self.model = MLPRegressor()
        self.model.fit(x, y)
        feature_select.get_feats(x, y)

    def test(self, docs, query=None):
        info = {}
        info['tf_idf'] = tf_idf.get_tf_idfs(docs)
        sents = []
        cluster_counts = llr.get_cluster_counts(docs)
        back_list, vocab = kl.get_freq_list(docs)
        for document in docs.keys():
            a_doc = docs[document]
            index = 0
            if len(a_doc) > 1:
                sents.append((a_doc[1], 1))
            else:
                sents.append((a_doc[0], 1))

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

                    vec = np.array(vec).reshape(1, -1)
                    vec = self.scaler.transform(vec)

                    # Add additional features here
                    # position_mul = math.fabs(0.5 - float(index) / len(a_doc))
                    sents.append((sentence, self.model.predict(vec)))
        return sorted(sents, key=lambda x: x[1], reverse=True)
