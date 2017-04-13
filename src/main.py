import json
import ContentSelector

gold = json.load(open('data/training.human.json'))
documents = json.load(open('data/training.json'))

selector = ContentSelector.ContentSelector()
selector.train(documents, gold)

for event in documents.keys():
    an_event = documents[event]
    results = selector.test(an_event, 10)
    print(sorted(list(results.keys()), key=lambda x: results[x], reverse=True))
