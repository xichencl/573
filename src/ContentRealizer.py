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
        results = list();
        for sent in sentences:
            if total_len < length_limit:
                text = sent[0];

                if self.max_cosine_sim(text,results) < 1.0:
                    total_len += len(text.split());
                    results.append(text);
        return results;

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(item) for item in tokens]

    def normalize(self, text):
        return self.stem_tokens(nltk.word_tokenize(text.lower().translate(self.remove_punctuation_map)))

    #def cosine_sim(self, text1, text2):
    #    tfidf = self.vectorizer.fit_transform([text1, text2])
    #    return ((tfidf * tfidf.T).A)[0,1]

    def max_cosine_sim(self,text, results):
        if len(results) == 0 : return 0;

        _result = list(results)
        _result.insert(0,text)

        tfidf = self.vectorizer.fit_transform(_result)
        max_sim = 0;

        for i in range(1,len(_result)):
            sim = ((tfidf * tfidf.T).A)[0,i]
            if sim > max_sim:
                max_sim = sim;

        return max_sim;