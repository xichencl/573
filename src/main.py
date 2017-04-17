import json
import ContentSelector
import ContentSorter
import ContentRealizer
import sys
import re

gold = json.load(open(sys.argv[1], 'r'))
documents = json.load(open(sys.argv[2], 'r'))

selector = ContentSelector.ContentSelector()
selector.train(documents, gold)

realizer = ContentRealizer.ContentRealizer()
sorter = ContentSorter.ContentSorter();

for event in documents.keys():
    an_event = documents[event]
    results = selector.test(an_event, 10)
    sorted_result = sorter.sort(results);
    picked = realizer.realize(results, 100)
    summary = re.sub('\W+', ' ', ' '.join(picked))
    out = open('/Users/mackie/PycharmProjects/573/outputs/D2/' + event[:-1], 'w')
    out.write(summary)

