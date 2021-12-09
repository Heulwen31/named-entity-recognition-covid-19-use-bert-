import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch

def set_data(file, data):
    file = open(file, 'r')
    lines = file.readlines()
    ner_tag = []
    tokens = []
    for line in lines:
        temp_data = line.split(' ')
        if len(temp_data) == 2:
            tokens.append(temp_data[0])
            ner_tag.append(temp_data[1][:-1])
        else:
            data.append([tokens, ner_tag])
            tokens = []
            ner_tag = []

data_train = []
data_validation = []

set_data("/content/drive/MyDrive/Colab Notebooks/data/word/train_word.conll", data_train)
set_data('/content/drive/MyDrive/Colab Notebooks/data/word/test_word.conll', data_validation)


def appendAll(data):
    tokens = []
    entities = []   
    for i in range(len(data)):
        tokens.append(data[i][0])
        entities.append(data[i][1])
    return Dataset.from_pandas(pd.DataFrame({'tokens': tokens, 'ner_tags': entities}))




label_list = ['O','B-PATIENT_ID', 'I-PATIENT_ID', 'B-NAME', 'I-NAME', 'B-GENDER', 'I-GENDER', 'B-AGE', 'I-AGE', 'B-JOB', 'I-JOB', 'B-LOCATION', 'I-LOCATION', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-DATE', 'I-DATE', 'B-SYMPTOM_AND_DISEASE', 'I-SYMPTOM_AND_DISEASE', 'B-TRANSPORTATION', 'I-TRANSPORTATION']
label_encoding_dict = {'O': 0,'B-PATIENT_ID': 1, 'I-PATIENT_ID': 2, 'B-NAME': 3, 'I-NAME': 4, 'B-GENDER': 5, 'I-GENDER': 6, 'B-AGE': 6, 'I-AGE': 7, 'B-JOB': 8, 'I-JOB': 9, 'B-LOCATION': 10, 'I-LOCATION': 11, 'B-ORGANIZATION': 12, 'I-ORGANIZATION': 13, 'B-DATE': 14, 'I-DATE': 15, 'B-SYMPTOM_AND_DISEASE': 16, 'I-SYMPTOM_AND_DISEASE': 17, 'B-TRANSPORTATION': 18, 'I-TRANSPORTATION': 20}

task = "ner" 
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_dataset = appendAll(data_train)
test_dataset = appendAll(data_validation)
print(train_dataset[0])