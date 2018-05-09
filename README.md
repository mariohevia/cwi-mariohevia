# Complex Word Identification (CWI) Shared Task 2018 - Mario Hevia

## Getting started
To download the files and the datasets, you should clone the repository from the git bash prompt.

    https://github.com/mariohevia/cwi-mariohevia.git

## Runing the code

To run the code you should execute `python3 final.py` in the command line and it will print the results, it may take a while and use a lot of memory since it loads the word2vec model from GoogleNews-vectors (1.6 GB) or SBW-vectors (1.1 GB).

If you want to use the test set instead of the dev set you need to comment line 14 and uncomment line 15 from utils/dataset.py

## Files and folders

\datasets - there are saved the datasets as given for the project.

\freq_datasets - there are saved frequency datasets from the Leipzig Corpora Collection and a python file that preprocess the data \freq_datasets\freq_preprocess.py

\pretrained_models - there are the word2vec models for english and spanish

\utils - there are 6 python files from which only 5 are used when executing `python3 final.py` the other one (wordvecavg.py) was a test to use the average of word vectors to work with target words with more than one token.

\utils\baseline.py - python file with all the features (eventhough it's name is baseline here it's used in the improvement of the model

\utils\dataset.py - used to read from the datasets in \datasets

\utils\scorer.py - used to score the predictions of the models

\utils\syllable_spanish - python file used in the syllabisation of spanish words taken from [GitHub](https://github.com/mabodo/sibilizador/blob/master/Silabizator.ipynb)

\utils\wordvec - python file used as a baseline with word vectors
