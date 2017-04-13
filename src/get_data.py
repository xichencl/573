'''
Created on Apr 8, 2017

Extract main body texts in document sets from corpora on patas 

@author: xichentop
'''


"""
imported libraries
"""
import argparse
import codecs
import re
from lxml import etree, html  # @UnresolvedImport
import glob
import json


"""
Global variables
"""
ACQUAINT = '/corpora/LDC/LDC02T31'
ACQUAINT2 = '/corpora/LDC/LDC08T25/data' 


def extract_doc_paths(file_path):
    """
    return a dictionary mapping a topic title to a list of document titles   
    """
    parser = etree.XMLParser(recover=True)
    data_tree = etree.parse(codecs.open(file_path, encoding='iso-8859-1'), parser) 
    topics = data_tree.findall(".//topic")
    doc_titles = {}
    for t in topics:
        topic_id = t.attrib.get('id')
        file_names = [d.attrib.get('id') for d in t.findall('docsetA/doc')]
        doc_titles[topic_id]= file_names      
    for k, v in doc_titles.items():
        print(k)
        print(len(v), 'documents\n', v)
    return doc_titles    

def extract_articles(doc_titles, dir):
    """
    return a dict mapping topic title to a set of articles to be summarized 
    doc_sets: a dictionary {topic title: ([docsetA], [docsetB])}
    """
    files = {}
    
    for topic, dt in doc_titles.items():
        files[topic] = {}        
        for d in dt:
            if '_' in d:
#                 elem = d.rsplit('_', 1)
                folder = d[0:7].lower()
                file_name = d[0:14].lower()+'.xml' 
                file_path = '/'.join([ACQUAINT2, folder, file_name])
                f_in = codecs.open(file_path, 'rb')
                parser = etree.XMLParser(recover=True)
                data_tree = etree.parse(f_in, parser)
                docs = data_tree.xpath('.//DOC')
                for doc in docs:
                    doc_id = doc.attrib.get('id').strip()
                    if doc_id == d:
                        paragraphs = doc.xpath('.//TEXT//P|.//TEXT')
                        paragraphs = [e.text for e in paragraphs]
                        files[topic][doc_id] = paragraphs
                        
            else:
                file_path = '/'.join([ACQUAINT, d[0:3].lower(), d[3:7], d[3:11]+"*"])
                file_path = glob.glob(file_path)[0]
                f_in = codecs.open(file_path, 'rb')
                data_tree = html.parse(f_in) 
                docs = data_tree.xpath(".//doc")
                for doc in docs:
                    doc_id = doc.xpath(".//docno")
                    doc_id = doc_id[0].text.strip()
                    if doc_id == d:
                        paragraphs = doc.xpath('.//text/p|.//text')
                        paragraphs = [e.text for e in paragraphs]
                        files[topic][doc_id] = paragraphs
                        
                            
            f_in.close()
    
    for k, v in files.items():
        print(k)
        print(len(v), 'documents\n', v)
        
    with open(dir, 'w') as outfile:
        json.dump(files, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("extract texts")
    parser.add_argument("doc_path", help="path of doc sets to be extracted")
    parser.add_argument("dir", help="dir where the articles for summ will be saved")
    args = parser.parse_args()
    doc_titles = extract_doc_paths(args.doc_path)
    extract_articles(doc_titles, args.dir)
#     doc_titles = extract_doc_paths('C:/Users/xichentop/Documents/573/GuidedSumm10_test_topics.xml')
#     print(doc_titles)
    
#     extract_articles(doc_titles, 'C:/Users/xichentop/Documents/573/')
#     print(doc_sets)    
#     extrac_articles(doc_sets)
