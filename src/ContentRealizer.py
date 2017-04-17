import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentRealizer:
    def __init__(self):
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        self.vectorizer = TfidfVectorizer(tokenizer=self.normalize, stop_words='english')
        return

    #pick up input sentences by their scores
    #sentences: list of sentences sorted with each score
    #return: picked sentences
    def realize(self, sentences, length_limit):
        total_len = 0;
        result_indices = list();
        #convert sentences
        sentences = [row[0] for row in sentences];
        tfidf = self.vectorizer.fit_transform(sentences)

        for i in range(0,len(sentences)):
            #cannot exceed length_limit
            if total_len >= length_limit:
                break
            if self.max_cosine_sim(i,result_indices,tfidf) < 1.0:
                result_indices.append(i);
                total_len += len(sentences[i].split());

        results = list();
        for index in result_indices:
            results.append(sentences[index])
        return results;

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(item) for item in tokens]

    def normalize(self, text):
        return self.stem_tokens(nltk.word_tokenize(text.lower().translate(self.remove_punctuation_map)))

    #def cosine_sim(self, text1, text2):
    #    tfidf = self.vectorizer.fit_transform([text1, text2])
    #    return ((tfidf * tfidf.T).A)[0,1]
    def max_cosine_sim(self,text_index, result_indices, tfidf):
        if len(result_indices) == 0 : return 0;
        max_sim = 0;
        for i in range(0,len(result_indices)):
            #not count self
            if text_index == i: continue
            sim = (tfidf * tfidf.T).A[text_index, i]
            if sim > max_sim:
                max_sim = sim;

        return max_sim;