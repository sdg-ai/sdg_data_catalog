# Dataset NER

Query to parse papers, etxract metadata and store them to a database:

```python
python dataset_ner/paper_processing.py -p '../aiforgood/test' -t 16 -n 'papers_metadata_and_cadidates.jsonl' -c False
```

**Arguments**:
**-p**: path to access the folder that contains all the papers to parse
**-t**: number of threads to use
**-n**: Name of the generated file [depeciated]
**-c**: Bool to indicate whether or not a new database need to be created

Query to generate a prodigy ready jsonl input file:

```python
python dataset_ner/prodigy_processing.py -n 'papers_metadata_and_cadidates.jsonl' -t 16
```

**Arguments**:
**-t**: number of threads to use
**-n**: Name of the generated file, that will serve as input to Prodigy



