import json
import ContentSelector
import ContentRealizer
import ContentSorter
import sys
import re

gold = json.load(open(sys.argv[1], 'r'))
documents = json.load(open(sys.argv[2], 'r'))

selector = ContentSelector.ContentSelector()
selector.train(documents, gold)

sorter = ContentSorter.ContentSorter();

realizer = ContentRealizer.ContentRealizer()

index = 0
for event in documents.keys():
    index += 1
    an_event = documents[event]
    print("Testing " + str(index) + '/' + str(len(documents.keys())))
    results = selector.test(an_event, 10)
    sorter.sort(results);
    picked = realizer.realize(results, 100)
    summary = re.sub('\W+', ' ', ' '.join(picked))

    if 'Group' in event:
        out = open('/home2/mblac6/573/573/outputs/D2/' + event, 'w')
    else:
        out = open('/home2/mblac6/573/573/outputs/D2/' + event[:-1], 'w')
    out.write(summary)


