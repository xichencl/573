import json
import ContentSelector
import ContentSorter
import ContentRealizer
import InfoOrderer
import sys
import re

gold = json.load(open(sys.argv[1], 'r'))
documents = json.load(open(sys.argv[2], 'r'))

selector = ContentSelector.ContentSelector()
selector.train(documents, gold)

realizer = ContentRealizer.ContentRealizer()
realizer = ContentRealizer.ContentRealizer();

for event in documents.keys():
    an_event = documents[event]
    results = selector.test(an_event, 10)
    picked = realizer.realize(results, 100)
    summary = re.sub('\W+', ' ', ' '.join(picked))
    out = open('/Users/mackie/PycharmProjects/573/outputs/D2/' + event[:-1], 'w')
    out.write(summary)

