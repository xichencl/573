import nltk
import math
import json
from sklearn.linear_model import LinearRegression

class ContentSelector:
    def __init__(self):
        self.model = None

    # return a dictionary with the tf*idfs for a set of documents
    def get_tf_idfs(self, docs):
        tf_idfs = {}
        doc_counts = {}
        word_counts = {}

        for doc in docs.keys():
            a_doc = docs[doc]
            if len(a_doc) > 1:
                words = nltk.word_tokenize(a_doc[1])
            else:
                words = nltk.word_tokenize(a_doc[0])
            for word in words:
                if word in word_counts.keys():
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
            vocab = set(words)
            for term in vocab:
                if term in doc_counts.keys():
                    doc_counts[term] += 1
                else:
                    doc_counts[term] = 1
        for key in doc_counts.keys():
            tf_idfs[key] = word_counts[key] * math.log(float(len(docs)) / doc_counts[key])
        return tf_idfs

    def get_tf_idf_average(self, sent, tf_idfs):
        tf_idf_sum = 0
        words = nltk.word_tokenize(sent)
        for word in words:
            if word in tf_idfs.keys():
                tf_idf_sum += tf_idfs[word]
        return float(tf_idf_sum) / len(words)

    # place any code that needs access to the gold standard summaries here
    def train(self, docs, gold):

        # dictionary containing feature information for each document cluster
        cluster_info = {}
        for event in docs.keys():
            an_event = docs[event]
            tf_idfs = self.get_tf_idfs(an_event)
            cluster_info[event] = {}
            cluster_info[event]["tf_idf"] = tf_idfs

        # process sentences in each document of each cluster
        x = []
        y = []
        for event in cluster_info.keys():
            an_event = docs[event]
            for document in an_event.keys():
                a_doc = an_event[document][1]

                # construct a vector for each sentence in the document
                for sentence in a_doc.split('\n'):
                    if len(sentence):
                        vec = []
                        vec.append(self.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                        x.append(vec)
                        y.append(0)
            gold_sums = gold[event]
            for document in gold_sums.keys():
                a_sum = an_event[document][1]

                # construct a vector for each sentence in the summary
                for sentence in a_sum.split('\n'):
                    if len(sentence):
                        vec = []
                        vec.append(self.get_tf_idf_average(sentence, cluster_info[event]["tf_idf"]))
                        x.append(vec)
                        y.append(1)

        self.model = LinearRegression(x, y)

    def test(self, docs, compression):
        info = {}
        info['tf_idf'] = self.get_tf_idfs(docs)
        sents = {}
        for document in docs.keys():
            a_doc = docs[document][1]

            # construct a vector for each sentence in the document
            for sentence in a_doc.split('\n'):
                vec = []
                vec.append(self.get_tf_idf_average(sentence, info["tf_idf"]))
                sents[sentence] = self.model.predict(vec)
        return sents

