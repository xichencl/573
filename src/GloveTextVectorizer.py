import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

EMBEDDING_DIM = 50

class GloveTextVectorizer:
    #build vectorizer from glove embedding file
    def __init__(self, glove_file_location):
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        self.vectorizer = TfidfVectorizer(tokenizer=self.normalize, stop_words='english')

        print('Indexing word vectors.')
        self.embeddings_index = {}
        f = open(glove_file_location, encoding="utf-8")
        i = 0
        for line in f:
            try:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
            except ValueError as e:
                print ("error", e, "on line", i)
            i += 1
        f.close()
        print('Found %s word vectors.' % len(self.embeddings_index))

        return

    #given list of sentences, output their corresponding text vectors in a list
    def getTextEmbedding(self,sentences):
        tfidf = self.vectorizer.fit_transform(sentences)
        sentence_vectors = [];
        for sentence in sentences:
            sentence_coefs = np.zeros(EMBEDDING_DIM);
            words = sentence.split(' ');
            for word in words:
                if word.lower() in self.embeddings_index:
                    sentence_coefs = np.add(sentence_coefs, self.embeddings_index[word.lower()]);
            sentence_coefs = sentence_coefs/len(words)
            sentence_vectors.append(sentence_coefs);
        return sentence_vectors

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(item) for item in tokens]

    def normalize(self, text):
        return self.stem_tokens(nltk.word_tokenize(text.lower().translate(self.remove_punctuation_map)))