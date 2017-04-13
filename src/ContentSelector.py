import nltk
import json
import re
import tf_idf
import llr

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

import ling_features
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

class ContentSelector:
    def __init__(self):
        self.model = None
        self.background_counts = None

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
        for event in cluster_info.keys():
            an_event = docs[event]
            cluster_counts = llr.get_cluster_counts(an_event)
            for document in an_event.keys():
                a_doc = an_event[document]
                for sentence in a_doc:

                    # construct a vector for each sentence in the document
                    if len(sentence.split()) > 1:
                        vec = []
                        vec.extend(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                        vec.append(llr.get_weight_sum(sentence, back_counts, cluster_counts))
                        vec = ling_features.add_feats(an_event, sentence, vec)

                        # Add additional features here
                        x.append(vec)
                        y.append(0)
            gold_sums = gold[event]
            for document in gold_sums.keys():
                a_sum = gold_sums[document]
                a_sum = re.sub('\n', ' ', a_sum)

                # construct a vector for each sentence in the summary
                for sentence in nltk.sent_tokenize(a_sum):
                    if len(sentence) > 1:
                        vec = []
                        vec.extend(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                        vec.append(llr.get_weight_sum(sentence, back_counts, cluster_counts))
                        vec = ling_features.add_feats(an_event, sentence, vec)

                        # Add additional features here
                        x.append(vec)
                        y.append(1)

        self.model = LinearRegression()
        self.model.fit(x,y)

        x = np.asarray(x)
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)

        forest.fit(x, y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(x.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(x.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(x.shape[1]), indices)
        plt.xlim([-1, x.shape[1]])
        plt.show()

    def test(self, docs, compression):
        info = {}
        info['tf_idf'] = tf_idf.get_tf_idfs(docs)
        sents = {}
        cluster_counts = llr.get_cluster_counts(docs)
        for document in docs.keys():
            a_doc = docs[document]

            # construct a vector for each sentence in the document
            for sentence in a_doc:
                if len(sentence.split()) > 1:
                    vec = []
                    vec.extend(tf_idf.get_tf_idf_average(sentence, info["tf_idf"]))
                    vec.append(llr.get_weight_sum(sentence, self.background_counts, cluster_counts))
                    vec = ling_features.add_feats(docs, sentence, vec)

                    # Add additional features here
                    sents[sentence] = self.model.predict(vec)
        return sents

