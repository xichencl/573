import nltk
import json
import re
import tf_idf
import ling_features
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

class ContentSelector:
    def __init__(self):
        self.model = None

    # place any code that needs access to the gold standard summaries here
    def train(self, docs, gold):

        # dictionary containing feature information for each document cluster
        cluster_info = {}
        for event in docs.keys():
            an_event = docs[event]
            tf_idfs = tf_idf.get_tf_idfs(an_event)
            cluster_info[event] = {}
            cluster_info[event]["tf_idf"] = tf_idfs

        # process sentences in each document of each cluster
        x = []
        y = []
        for event in cluster_info.keys():
            an_event = docs[event]
            for document in an_event.keys():
                a_doc = an_event[document]
                for sentence in a_doc:

                    # construct a vector for each sentence in the document
                    if len(sentence.split()) > 1:
                        vec = []
                        vec.append(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
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
                        vec.append(tf_idf.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                        vec = ling_features.add_feats(an_event, sentence, vec)

                        # Add additional features here
                        x.append(vec)
                        y.append(1)

        self.model = LinearRegression()
        self.model.fit(x,y)

    def test(self, docs, compression):
        info = {}
        info['tf_idf'] = tf_idf.get_tf_idfs(docs)
        sents = {}
        for document in docs.keys():
            a_doc = docs[document]

            # construct a vector for each sentence in the document
            for sentence in a_doc:
                if len(sentence.split()) > 1:
                    vec = []
                    vec.append(tf_idf.get_tf_idf_average(sentence, info["tf_idf"]))
                    vec = ling_features.add_feats(docs, sentence, vec)

                    # Add additional features here
                    sents[sentence] = self.model.predict(vec)
        return sents

