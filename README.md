# Causal-Debias
This repository provides a reference implementation of paper: 
*Causal-Debias: Unifying Debiasing in Pretrained Language Models and Fine-tuning via Causal Invariant Learning*.

## Requirments
We implement Causal-Debias with following dependencies:
* tqdm 
* torch
* regax
* numpy
* functools
* scikit-learn
* transformers

## Data
Download the GLUE data (SST-2, QNLI, CoLA). After downloading, unpack it to some directory.

## External Corpora
We use the Corpora from the paper [Towards Debiasing Sentence Representations](https://aclanthology.org/2020.acl-main.488/)

## Usage

Go to directory `experiment`.

### A. Preprocess the external corpora

```
python preprocess.py \
--model_type   bert or roberta or albert
--model_name_or_path  bert-base-uncased, etc
--no_cuda  if to use cuda
--cuda  if use cuda, choose which cuda to use
--max_seq_length  The maximum total input sequence length after WordPiece  tokenization
--output  which file to save the preprocess embedding temporarily their is a  default path
```

### B. Debiasing language models while fintuning
```
python causal-debias.py
--debias_type  gender or race
--model_type   bert or roberta or albert
--model_name_or_path  bert-base-uncased, etc
--prompts_file prompts_bert-base-uncased_gender
--task_name  SST-2 QNLI or CoLA
--data_dir  the path that saved glue data
--no_cuda  if to use cuda
--cuda  if use cuda, choose which cuda to use
--k  top k similar sentence to extend
--tau  a trade-off hyperparameters
```
