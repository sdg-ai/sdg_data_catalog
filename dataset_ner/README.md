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



