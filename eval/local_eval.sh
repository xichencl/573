#!/usr/bin/env bash
python3 ../src/main.py ../src/data/new_human.processed.json ../src/data/new_training.processed.json ../src/data/training.json ../src/data/training.processed.json
./loc_ROUGE-1.5.5.pl -e data -a -n 4 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d /Users/mackie/PycharmProjects/573/eval/local_tr.xml