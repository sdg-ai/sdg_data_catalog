Created by Surnjani Djoko, Jan 15 2022
# Topic modeling
 1. TM_1_DataCleaning notebook performs data cleaning and data preparation such as lower case all words, clear texts from punctuation, remove stopwords, remove words with a length below 3 characters, Lemmatize words, prune high frequency words, etc.
 2. TM_2_Modeling notebook used the cleaned/prep dataset to create bigrams, and corpus, and feed into LDA. With the small sample dataset, it seems 8 topics was the best option.

# Limitations
 1. It is tested with a small sample size
 2. The data input (paper abstracts) have mixed languages besides English (some Indonesian abstracts were found and had been manually removed)
 3. Hard to interpret the LDA results, some topics contain irrelevant words
 


 
