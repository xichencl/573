import nltk

def get_positions(docs):
    first_positions = {}
    all_positions = {}
    for doc in docs:
        seen = set()
        a_doc = docs[doc]
        words = nltk.word_tokenize(' '.join(a_doc))
        index = 0
        for word in words:
            if word not in nltk.corpus.stopwords.words('english'):
                if word not in first_positions.keys():
                    first_positions[word] = []
                if word not in all_positions.keys():
                    all_positions[word] = []
                if word not in seen:
                    first_positions[word].append(index)
                    seen.add(word)
                all_positions[word].append(index)
            index += 1
    return first_positions, all_positions


def score_sent(sent, first, all):
    data = [0, 0, 0]
    words = nltk.word_tokenize(sent)
    for word in sent:
        if word in first.keys() and word in all.keys():
            data[0] += float(min(first[word]))
            data[1] += float(sum(first[word])) / len(first[word])
            data[2] += float(sum(all[word])) / len(all[word])
    data[0] /= len(words)
    data[1] /= len(words)
    data[2] /= len(words)
    return data