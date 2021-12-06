# Dataset NER

### Data preprocessing

Query to parse papers, etxract metadata and store them to a database:

```python
python dataset_ner/paper_processing.py -p '../aiforgood/test' -t 16 -n 'papers_metadata_and_cadidates.jsonl' -c False
```

**Arguments**:<br/>
**-p**: path to access the folder that contains all the papers to parse<br/>
**-t**: number of threads to use<br/>
**-n**: Name of the generated file [depeciated]<br/>
**-c**: Bool to indicate whether or not a new database need to be created<br/>

Query to generate a prodigy ready jsonl input file:

```python
python dataset_ner/prodigy_processing.py -n 'papers_metadata_and_cadidates.jsonl' -t 16
```

**Arguments**:<br/>
**-t**: number of threads to use<br/>
**-n**: Name of the generated file, that will serve as input to Prodigy<br/>

### NER pipeline

To run the NER pipeline, the wrapper_ner.py script can be used.<br/>
**Pipeline**: Word Embedding (Glove) --> BiLSTM-CRF training --> Model evaluation --> Active Learning 

Query to train a new NER model:

```python
python dataset_ner/wrapper_ner.py -p 'db/annotations/annotations.csv' -t 16 -s True -a False -rn 'dataset_ner/performances/report_1.json'
```

**Arguments**:<br/>
**-p**: path of the training dataset<br/>
**-t**: number of threads to use<br/>
**-s**: boolean to indicate weather or not to save the performance values<br/>
**-a**: boolean to indicate weather or not we want to use Active Learning<br/>
**-rn**: name of the report file with the performance summary<br/>

Query to use Active Learning in order to identify the next data point to annotate from an unnanotated dataset:

```python
python dataset_ner/wrapper_ner.py -p 'db/annotations/annotations.csv' -t 16 -s True -a True -rn 'dataset_ner/performances/report_1.json' -as 'LTP' -bf .05 -ut "db/sdg_data_catalog_test.db" -na 'db/annotations/AL_generated_data_to_annotate.jsonl'
```

**Arguments**:<br/>
**-a**: boolean to indicate weather or not we want to use Active Learning<br/>
**-as**: which Active Learning strategy do we want to use (LTP, RANDOM, MTP...) <br/>
**-bf**: the batch fraction of annotated data that we want to annotate <br/>
**-ut**: the path to the unnanotated datset <br/>
**-na**: name of the prodigy ready generated file with the list of data to annotate as suggested by the Active Learning

## Current performances on the NER task:

This result are based on a 5k large corpus of papers.

| Model  | Recall | Precision  | F1 score |
| ---------- | ----------- | ----------- | ----------- |
| Glove 400k + BiLSTM CRF  |  0.78  | 0.6  | 0.68  |
| Glove 1.9M + BiLSTM CRF  | **0.88**  | 0.74  | 0.80 |
| BERT base uncased | 0.75  | **0.91**  | **0.82** |
