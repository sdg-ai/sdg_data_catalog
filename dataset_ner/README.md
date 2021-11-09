# Dataset NER

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

## Current performances on the NER task:

This result are based on a 5k large corpus of papers.

| Model  | Recall | Precision  | F1 score |
| ---------- | ----------- | ----------- | ----------- |
| Glove 400k + BiLSTM CRF  |  0.78  | 0.6  | 0.68  |
| Glove 1.9M + BiLSTM CRF  | **0.88**  | 0.74  | 0.80 |
| BERT base uncased | 0.75  | **0.91**  | **0.82** |
