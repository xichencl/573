import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import GloveTextVectorizer
import LabelMarker
import re
import SentenceCompressor


class ContentRealizer:
    def __init__(self):
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.normalize, stop_words='english')
        self.glove_vectorizer = GloveTextVectorizer.GloveTextVectorizer("../src/word_embedding/glove.6B.50d.txt")

        self.TEXT                 = 0
        self.SCORE                = 1
        self.ARTICLE              = 2
        self.NTH_SENT             = 3
        self.PUB_DATE             = 4
        self.GLOBAL_SENT_ID       = 5
        self.SENT_GROUP_ID        = 6
        return

    # pick up input sentences by their scores
    # sentences: output of content selector
    # return: picked sentences
    def realize(self, _sentences, length_limit):
        # convert sentences
        # id_sentence_map ->  {global_sent_id : [text, score, article_name, nth_sentence, publish_date, global_sent_id, sentence_group_idx]}
        id_sentence_map  = self.convert_sentences(_sentences)

        # sort sentences desc according to score
        sentences = list(id_sentence_map.values())
        sentences = sorted(sentences, key=lambda x: x[1], reverse=True)
        tfidf = self.tfidf_vectorizer.fit_transform([s[0] for s in sentences])

        # pick up k sentences based on top scores
        result_global_indices = self.pick_sentences(sentences, tfidf, length_limit = 100)
        # sentence embeddings
        sentence_vectors = self.glove_vectorizer.getTextEmbedding([s[0] for s in sentences])

        # label propagation
        marker = LabelMarker.LabelMarker()
        labels = [-1]*len(sentence_vectors)
        #set k seed indices
        for i in range(0, len(result_global_indices)):
            labels[i] = i
        #clustering labels
        labels = marker.label(sentence_vectors, labels)
        for i in range(0,len(labels)):
            id_sentence_map[sentences[i][self.GLOBAL_SENT_ID]][self.SENT_GROUP_ID] = labels[i]

        #C[i,j] = freq(i,j)^2 / (freq(i)* freq(j))
        adjacent_matrix = self.calc_adjacent_matrix(id_sentence_map, labels, len(result_global_indices), window_size = 2)

        #greedy sorting by adjacency
        result = [id_sentence_map[idx] for idx in result_global_indices]
        #the first sentence in the first passage as boosting
        #TODO: replace with groups with most BOS adjacent
        result.sort(key = lambda x: (x[self.PUB_DATE],x[self.GLOBAL_SENT_ID]))

        for i in range(1,len(result)-1):
            #   then pick arg mx(C[i,j])
            class_idx1 = result[i-1][self.SENT_GROUP_ID]
            # sort by group prob
            result[i:] = sorted(result[i:], key=lambda x: adjacent_matrix[class_idx1][x[self.SENT_GROUP_ID]], reverse=True)

        return [SentenceCompressor.compress(s[0]) for s in result]

    # C[i,j] = freq(i,j)^2 / (freq(i)* freq(j))
    # article_sentence_map  ->  {article_name: [global_sentence_id[0], global_sentence_id[1], ...]}
    # id_sentence_map       ->  {global_sent_id : [text, score, article_name, nth_sentence, publish_date, global_sent_id, sentence_group_idx]}
    def calc_adjacent_matrix(self, id_sentence_map, labels, categories_of_labels, window_size):

        #freq(i)
        uniclass_freq = [0] * categories_of_labels
        for idx in range(0,len(labels)):
            uniclass_freq[labels[idx]] += 1

        #freq(i,j)
        import numpy as np
        biclass_freq = np.array([[0]*categories_of_labels]*categories_of_labels)
        id_sentence_list = list(id_sentence_map.items())
        id_sentence_list = sorted(id_sentence_list, key=lambda x: x[0], reverse=False)

        for label_idx in range(0,categories_of_labels):
                for i in range(0,len(id_sentence_list)):
                    if id_sentence_list[i][1][self.SENT_GROUP_ID] == label_idx:
                        id_sentence_slice = id_sentence_list[i-window_size:i+window_size+1]
                        for id_sentence_tuple in id_sentence_slice:
                            #same article adjacent, not the same sentence
                            if id_sentence_tuple[1][self.ARTICLE] == id_sentence_list[i][1][self.ARTICLE] \
                                    and id_sentence_tuple[1][self.GLOBAL_SENT_ID] != id_sentence_list[i][1][self.GLOBAL_SENT_ID]:
                                label_idx2 = id_sentence_tuple[1][self.SENT_GROUP_ID]
                                biclass_freq[label_idx][label_idx2] += 1

        #C[i,j] = freq(i,j)^2 / (freq(i)* freq(j))
        adjacent = np.array([[0.0] * categories_of_labels] * categories_of_labels)
        for i in range(0,categories_of_labels):
            for j in range(0, categories_of_labels):
                adjacent[i][j] = float((biclass_freq[i][j]*biclass_freq[i][j]))/float((uniclass_freq[i]*uniclass_freq[j]))

        return adjacent

    def pick_sentences(self, sentences, tfidf, length_limit):
        result_indices = []
        total_len = 0

        for i in range(0, len(sentences)):
            # cannot exceed length_limit
            if total_len >= length_limit:
                break
            if self.max_cosine_sim(i, result_indices, tfidf) < 0.4:
                result_indices.append(i)
                compressed = SentenceCompressor.compress(sentences[i][0])
                total_len += len(compressed.split())
        #get global idx
        global_result_indices = [sentences[idx][self.GLOBAL_SENT_ID] for idx in result_indices]
        return global_result_indices

    # id_sentence_map       ->  {global_sent_id : [text, score, article_name, nth_sentence, publish_date, global_sent_id, sentence_group_idx]}
    def convert_sentences(self, _sentences):
        id_sentence_map = {}

        global_index = 0
        for article_name in _sentences:
            sentence_list = _sentences[article_name]
            publish_date = re.search(r"(\d+)", article_name).group()
            for i in range(0, len(sentence_list)):
                sentence_tuple = sentence_list[i]
                id_sentence_map[global_index] = [sentence_tuple[0], sentence_tuple[1], article_name, i, int(float(publish_date)), global_index, -1]
                global_index += 1

        return id_sentence_map

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(item) for item in tokens]

    def normalize(self, text):
        return self.stem_tokens(nltk.word_tokenize(text.lower().translate(self.remove_punctuation_map)))

    def max_cosine_sim(self, text_index, result_indices, tfidf):
        if len(result_indices) == 0: return 0
        max_sim = 0
        for i in range(0, len(result_indices)):
            # not count self
            if text_index == i: continue
            sim = (tfidf * tfidf.T).A[text_index, i]
            if sim > max_sim:
                max_sim = sim

        return max_sim
