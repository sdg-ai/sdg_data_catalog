import sys
#sys.path.append("add location to the root folder if needed")
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#print(sys.path)
try:
    from ner_pipeline import *
    from prodigy_processing import *
except: 
    from dataset_ner.ner_pipeline import *
    from dataset_ner.prodigy_processing import *
import argparse
import functools

def wrapper(training_data_path='db/annotations/annotations.csv', 
            model_to_train='bert', scores=True, active_learning=False, active_learning_strategy='LTP',
            batch_fraction=.05, unannotated_texts_path='db/annotations/annotations.csv', generated_file_path='db/annotations/AL_generated_data_to_annotate.jsonl'):
    T = NER_AND_AL_Pipeline(training_data_path=training_data_path, model_to_train=model_to_train)
    T.word_embedding()
    T.build_bilstm_crf()
    entity_f1_score,sentence_ac_score = T.predict_eval()
    if scores:
        #save the scores
        print('TODO')
    else: pass
    if active_learning:
        dat = pd.DataFrame(extract_all_paragraphs(database=unannotated_texts_path), columns=['paragraph_id', 'body_text', 'paper_id'])
        unannotated_texts = dat['body_text']
        unannotated_labels = [0 for i in range(len(unannotated_texts))]
        res = T.active_learning(unannotated_texts, unannotated_labels, 
                        strategy=active_learning_strategy, query_batch_fraction=batch_fraction, 
                        ret_idx=True)
        data_to_annotate = dat.loc[res]
        list(map(functools.partial(process_paragraph, generated_file_path=generated_file_path), list(data_to_annotate.values)))
    return

if __name__ == '__main__':
    '''
    serialization of the pipeline to train and evaluate the NER model
    '''
    prs = argparse.ArgumentParser()
    prs.add_argument('-p', '--path_annotations', help='Path to access the directory with all annotations', type=str, default='db/annotations/annotations.csv')
    prs.add_argument('-t', '--num_threads', help='Number of Threads to use', type=int, default=8)
    prs.add_argument('-m', '--model', help='Which model to train?', type=str, default='bert')
    prs.add_argument('-s', '--scores', help='if true, train a model on the annotations and save the scores', type=bool, default=True)
    prs.add_argument('-a', '--active_learning', help='if true, save the next most informative data point to annotate', type=bool, default=True)
    prs.add_argument('-as', '--active_learning_strategy', help='which strategy do we want to use for the active learning', type=str, default='LTP')
    prs.add_argument('-bf', '--batch_fraction', help='the fraction of unnanotated data that we want to suggest for annotation', type=float, default=.05)
    prs.add_argument('-ut', '--unnanotated_text_path', help='Path to access the unnanotated data', type=str, default="db/sdg_data_catalog_test.db")
    prs.add_argument('-na', '--new_annotations', help='path of the newly created file with recommended data to annotate', type=str, default='db/annotations/AL_generated_data_to_annotate.jsonl')
    prs = prs.parse_args()
    wrapper(training_data_path=prs.path_annotations, model_to_train=prs.model,
    scores=prs.scores, active_learning=prs.active_learning, active_learning_strategy=prs.active_learning_strategy,
    batch_fraction=prs.batch_fraction, unannotated_texts_path=prs.unnanotated_text_path)
