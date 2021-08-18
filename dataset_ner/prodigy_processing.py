#import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import srsly
import json
import argparse
import jsonlines
from multiprocessing import Pool

from utils.db_connection import *

generated_file_path = 'prodigy_serialized_file.jsonl'
#name of the sqlite db created
database = r"db/sdg_data_catalog_test.db"

def prodigy_process():
    '''
    Function used to store all the selected paragraphs in a single jsonl file that can be used by Prodigy
    Args:
        file_path: path of the json with all paragraphs stored
    return the list of paragraph_ids
    '''
    def extract_all_paragraphs():
        conn = create_connection(database)
        cur = conn.cursor()
        cur.execute('''select paragraph_id, body_text, paper_id from paragraph''')
        res = cur.fetchall()
        return res
        
    def process_paragraph(prg_dat=str):
        '''
        Function used to add one paragraph to the jsonl file 
        Args:
            prg: the paragraph
        '''
        d = {}
        d['text'] = prg_dat[1]
        d['meta'] = {'paragraph_id':prg_dat[0], 'paper_id':prg_dat[2]}
        assert srsly.is_json_serializable([d]) is True
        srsly.write_jsonl(generated_file_path, [d], append=True, append_new_line=False)
        
    #can parallelize with multiprocessing for more efficiency
    list_paragraphs = extract_all_paragraphs()
    list(map(process_paragraph, list_paragraphs))
    return


if __name__ == '__main__':
    prs = argparse.ArgumentParser()
    prs.add_argument('-t', '--num_threads', help='Number of Threads to use', type=int, default=8)
    prs.add_argument('-n', '--file_name', help='Name of the generated file', type=str, default='papers_metadata_and_cadidates.jsonl')
    prs = prs.parse_args()

    generated_file_path = prs.file_name
    jsonlines.open(prs.file_name, mode='w')
    print('starting the script')
    prodigy_process()