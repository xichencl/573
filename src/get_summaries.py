'''
Created on Apr 11, 2017
extract human and peer generated summaries from patas

@author: xichentop
'''
import argparse
import codecs
import os
import json

def extract_docs(input_dir, output_dir):
    summaries = {}
    for filename in os.listdir(path=input_dir):
        topic_id = filename[0:5]+filename[-3]
        if topic_id not in summaries:
            summaries[topic_id] = {}
        with codecs.open(input_dir+'/'+filename, 'r') as f:
            summaries[topic_id][filename] = f.read()
        
    with codecs.open(output_dir, 'w') as outfile:
        json.dump(summaries, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("extract human summaries")
    parser.add_argument("doc_path", help="path of doc sets to be extracted")
    parser.add_argument("dir", help="dir where the docs will be saved")
    args = parser.parse_args()
    extract_docs(args.doc_path, args.dir)
