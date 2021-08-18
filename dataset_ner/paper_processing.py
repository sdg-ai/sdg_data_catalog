#import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.fastmatch import *
from utils.db_connection import *

import glob 
import random
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import argparse
import jsonlines
#NLP
import spacy
from bs4 import BeautifulSoup
nlp = spacy.load("en_core_web_sm")

from multiprocessing import Pool
import sys 
import uuid
sys.setrecursionlimit(10**6) #Increment recursion limits

#name of the sqlite db created
database = r"db/sdg_data_catalog_test.db"

def create_db_tables():
    # create a database connection
    conn = create_connection(database)
    # create tables
    create_table(conn, create_papers_table)
    create_table(conn, create_paragraph_table)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print('list of tables: ', cursor.fetchall())
    return


class Paper():

    '''
    Class for papers with the processing steps to extract paragraph and metadata, and identify candidates paragraphs for the NER task
    
    init takes one argument: the path to access the paper we are processing
    the abstract, title, paragraphs, authors and affiliation functions are used to extract metadata from the paper
    '''
    
    def __init__(self, path=str):
        self.path = path 
        self.paragraph_list = []
        self.contents = None
        self.pTag = None
        self.authors = []
        self.affiliations = []
        #TODO: extract DOI
        self.doi = ''
        with open(self.path, 'r') as f:
            self.data = f.read()
        self.bs = BeautifulSoup(self.data, 'xml')
    
    def get_abstract(self):
        if not self.bs.select("abstract"):
            self.abstract = None
        else:
            self.abstract = self.bs.abstract.contents[1].string
        return self.abstract
        
    def get_title(self, min_words = int):
        """ Extract the Text associated to the "article-title"
        tag, and evaluates if the number of tokens in the text is bigger than
        an estimated average"""
        
        if not self.bs.select("article-title"):
            self.title = None
        else:
            if len(str(self.bs.select("article-title")[0].string).split(' ')) >= min_words:
                self.title = str(self.bs.select("article-title")[0].string)
            else:
                self.title = None
        return self.title

    def get_paragraphs(self, ret):
        """ Extract the text associated to the tag <p> paragraphs"""
        self.pTag = self.bs.select("p")
        self.paragraphs_list = [el.text.replace('\n', ' ').replace('  ', ' ').strip() for el in self.pTag]
        if ret:
            return self.paragraphs_list

    def get_authors(self):
        """ Extract the text associated to the tag <contrib>"""
        if not self.bs.select('contrib'):
            self.authors = None
        else:
            querry = self.bs.find_all("contrib", {"contrib-type":"author"})
            for i in range(len(querry)):
                self.authors.append(querry[i].get_text())
        return self.authors
        
    def get_affiliation(self):
        if not self.bs.select('institution'):
            self.affiliations = None
        else:
            institutions = self.bs.find_all("institution")
            for i in range(len(institutions)):
                self.affiliations.append(institutions[i].get_text())
        return self.affiliations

            
def wrapper_paper(path):
    '''
    Different steps to process papers and save the metadata and candidate paragraphs to a sqlite db
    '''
    pp = Paper(path)
    pp.get_paragraphs(ret=False)
    pp.get_abstract()
    pp.get_title(min_words=3)
    pp.get_authors()
    pp.get_affiliation()

    #pm = FastMatch()
    #for i in getattr(pp,'paragraphs_list'):
#
    #    pm.match(i.lower())
    rd = random.Random()
    rd.seed(getattr(pp,'path'))
    d_paper = {
        'paper_id':str(uuid.UUID(int=rd.getrandbits(128))),
        'title': getattr(pp,'title'),
        'paper_path':getattr(pp,'path'),
        #TODO
        'doi':'',
        #TODO
        'date':'',
        'authors':str(getattr(pp,'authors')),
        'affiliations':str(getattr(pp, 'affiliations')),
        #TODO: update it with narrower candidates
        #'paragraphs_candidates':str([el for el in getattr(pp, 'paragraphs_list') if 'data' in el]),
        #'dataset_names':getattr(pm,'dataset_name'),
        #'paragraphs_total':getattr(pp, 'paragraphs_list')
    }
    #assert srsly.is_json_serializable([d_paper]) is True
    #srsly.write_jsonl(generated_file_path, [d_paper], append=True, append_new_line=False)
    conn = create_connection(database)
    add_paper(conn, d_paper)
    for i, el in enumerate(getattr(pp, 'paragraphs_list')):
        if 'data' in el:
            rdp = random.Random()
            rdp.seed(getattr(pp,'path')+str(i))
            add_paragraph(conn, {'paragraph_id':str(uuid.UUID(int=rdp.getrandbits(128))), 'paper_id':d_paper['paper_id'], 'body_text':el})

if __name__ == '__main__':
    '''
    serialization of the processing of papers
    it takes three inputs: 
    - path: this is the path to access the directory that contains all papers that we want to process
    - num_threads: the number of threads that we want to use for the parallelization
    - file_name: the name of the file that we are creating (depreciated for now as we are using a sqlite db)
    '''
    prs = argparse.ArgumentParser()
    prs.add_argument('-p', '--path', help='Path to access the directory with all papers', type=str, default='xml/')
    prs.add_argument('-t', '--num_threads', help='Number of Threads to use', type=int, default=8)
    prs.add_argument('-n', '--file_name', help='Name of the generated file', type=str, default='papers_metadata_and_cadidates.jsonl')
    prs.add_argument('-c', '--create_db', help='bool to indicate whether or not the db needs to be created', type=bool, default=False)
    prs = prs.parse_args()

    if prs.create_db:
        create_db_tables()
    generated_file_path = prs.file_name
    jsonlines.open(prs.file_name, mode='w')
    p = Pool(prs.num_threads)
    print('starting the script')
    list_of_paper_path = glob.glob('{}/*.cermxml'.format(prs.path))
    print('Processing ', len(list_of_paper_path), 'number of papers')
    p.map(wrapper_paper, list_of_paper_path)