import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import GloveTextVectorizer
import LabelMarker
import itertools


class ContentRealizer:
    def __init__(self):
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.normalize, stop_words='english')
        self.glove_vectorizer = GloveTextVectorizer.GloveTextVectorizer("./word_embedding/glove.6B.50d.txt")
        return

    # pick up input sentences by their scores
    # sentences: output of content selector
    # return: picked sentences
    def realize(self, _sentences, length_limit):
        # convert sentences
        # article_sentence_map  ->  {article_name: [global_sentence_id[0], global_sentence_id[1], ...]}
        # id_sentence_map       ->  {global_sent_id : [text, score, article_name, nth_sentence, publish_date, global_sent_id, sentence_group_idx]}
        article_sentence_map,id_sentence_map  = self.convert_sentences(_sentences)

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
            id_sentence_map[sentences[i][5]][6] = labels[i]


        #TODO: C[i,j] = freq(i,j)^2 / (freq(i)* freq(j))
        co_matrix = self.calc_adjacent_matrix(article_sentence_map, id_sentence_map, labels, window_size = 5)


        #always pick the first sentence of first article
        #then pick arg mx(C[i,j])

        results = list()
        for index in result_global_indices:
            results.append(sentences[index][0])
        return results

    # C[i,j] = freq(i,j)^2 / (freq(i)* freq(j))
    # article_sentence_map  ->  {article_name: [global_sentence_id[0], global_sentence_id[1], ...]}
    # id_sentence_map       ->  {global_sent_id : [text, score, article_name, nth_sentence, publish_date, global_sent_id, sentence_group_idx]}
    def calc_adjacent_matrix(self, article_sentence_map, id_sentence_map, labels, window_size):

        #freq(i)
        uniclass_freq = [0]*len(labels)
        for idx in range(0,len(labels)):
            uniclass_freq[labels[idx]] += 1
        uniclass_freq = [f/len(labels) for f in  uniclass_freq]

        #freq(i,j)
        biclass_freq = [[0] * len(labels)] * len(labels)
        article_sentence_list = list(article_sentence_map.items())
        article_sentence_list = sorted(article_sentence_list, key=lambda x: x[0], reverse=True)
        id_sentence_list = list(id_sentence_map.items())
        id_sentence_list = sorted(id_sentence_list, key=lambda x: x[0], reverse=True)

        for label_idx in labels:
            for article in article_sentence_list:
                for i in range(0,article):
                    sentence_id = article[i]
                    if id_sentence_list[sentence_id][6] == label_idx:
                        sentence_id_slice = article[i-window_size:i+window_size+1]
                        for idx in sentence_id_slice:
                            biclass_freq[label_idx][idx] += 1
                            biclass_freq[idx][label_idx] += 1

        #C[i,j] = freq(i,j)^2 / (freq(i)* freq(j))
        adjacent = [[0] * len(labels)] * len(labels)
        for i in range(0,len(labels)):
            for j in range(0, len(labels)):
                adjacent[i][j] = (biclass_freq[i][j]*biclass_freq[i][j])/(uniclass_freq(i)*uniclass_freq(j))

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
                total_len += len(sentences[i][0].split())

        result_indices = [sentences[5] for idx in result_indices]
        return result_indices

    # article_sentence_map  ->  {article_name: [global_sentence_id[0], global_sentence_id[1], ...]}
    # id_sentence_map       ->  {global_sent_id : [text, score, article_name, nth_sentence, publish_date, global_sent_id, sentence_group_idx]}
    def convert_sentences(self, _sentences):
        article_sentence_map = {}
        id_sentence_map = {}

        global_index = 0
        for article_name in _sentences:
            sentence_list = _sentences[article_name]
            for i in range(0, len(sentence_list)):
                sentence_tuple = sentence_list[i]
                if article_name not in article_sentence_map:
                    article_sentence_map[article_name] = list()
                article_sentence_map[article_name].append(global_index)
                id_sentence_map[global_index] = (sentence_tuple[0], sentence_tuple[1], article_name, i, int(float(article_name.split('_')[2])), global_index)
                global_index += 1

        return (article_sentence_map,id_sentence_map)

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
