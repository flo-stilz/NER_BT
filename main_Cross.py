# huggingface transformers
# setup 2

import json
import numpy as np
import nltk
from seqeval.metrics import f1_score, classification_report
from seqeval.metrics.v1 import precision_recall_fscore_support
from seqeval.scheme import BILOU
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from transformers import DistilBertTokenizerFast, DistilBertTokenizer, DistilBertForTokenClassification, Trainer, TrainingArguments, BertTokenizerFast, BertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification
import torch
from sklearn.metrics import accuracy_score#, precision_recall_fscore_support, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import random
import time
import copy
import os
from datasets import load_dataset, load_metric
from typing import Dict


for line in open('data/texts.json', 'r'):
    texts = json.loads(line)
for line in open('data/tags.json', 'r'):
    tags = json.loads(line)


def encode_tags2(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset, doc_input_ids in zip(labels, encodings.offset_mapping, encodings.input_ids):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        doc_enc_labels[0] = tag2id['O']
        last_index = doc_input_ids.index(102)
        doc_enc_labels[last_index] = tag2id['O']
        for i in range(last_index):
            if doc_enc_labels[i] == -100:
                if 'B-' in id2tag[doc_enc_labels[i-1]]:
                    doc_enc_labels[i] = tag2id[id2tag[doc_enc_labels[i-1]].replace('B-','I-')]
                elif 'L-' in id2tag[doc_enc_labels[i-1]]:
                    doc_enc_labels[i-1] = tag2id[id2tag[doc_enc_labels[i-1]].replace('L-','I-')]
                    doc_enc_labels[i] = tag2id[id2tag[doc_enc_labels[i-1]].replace('I-','L-')]
                elif 'U-' in id2tag[doc_enc_labels[i-1]]:
                    doc_enc_labels[i-1] = tag2id[id2tag[doc_enc_labels[i-1]].replace('U-','B-')]
                    doc_enc_labels[i] = tag2id[id2tag[doc_enc_labels[i-1]].replace('B-','L-')]
                else:
                    doc_enc_labels[i] = tag2id[id2tag[doc_enc_labels[i-1]]]
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels 

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p = [[id2tag[p] for p in pre] for pre in preds]
    id2tag[-100] = '-100'# requires to be fixed!
    l = [[id2tag[p] for p in pre] for pre in labels]
    
    r = []
    for i in range(len(l)):
        for j in range(len(l[i])):
            if l[i][j] == '-100':
                r.append((i,j))
    r.reverse()
    for (i,j) in r:
        del p[i][j]
        del l[i][j]
    
    #result = classification_report(y_true=l, y_pred=p, scheme=BILOU, mode='strict')
    #print(result)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=l, y_pred=p, average="weighted", scheme=BILOU)
    
    return {
            'precision': precision,
            'recall': recall,
            'f1': f1,}
    
def model_init():
    return AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=len(unique_tags), return_dict=True)

def fix_tag_problems(tags, texts):
    # removes sentences with overlapping entities
    # should only occur due to problems within the data
    rem = []
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            if tags[i][j]=='-':
                rem.append(i)
    rem.reverse()
    for obj in rem:
        del tags[obj]
        del texts[obj]
    return tags, texts
    

# main:

for k in range(5):
    if k==0:
        for line in open('final_results_data/train_texts1.json', 'r'):
            train_texts = json.loads(line)
        for line in open('final_results_data/train_tags1.json', 'r'):
            train_tags = json.loads(line)
        for line in open('final_results_data/test_texts1.json', 'r'):
            val_texts = json.loads(line)
        for line in open('final_results_data/test_tags1.json', 'r'):
            val_tags = json.loads(line)
    elif k==1:
        for line in open('final_results_data/train_texts2.json', 'r'):
            train_texts = json.loads(line)
        for line in open('final_results_data/train_tags2.json', 'r'):
            train_tags = json.loads(line)
        for line in open('final_results_data/test_texts2.json', 'r'):
            val_texts = json.loads(line)
        for line in open('final_results_data/test_tags2.json', 'r'):
            val_tags = json.loads(line)
    elif k==2:
        for line in open('final_results_data/train_texts3.json', 'r'):
            train_texts = json.loads(line)
        for line in open('final_results_data/train_tags3.json', 'r'):
            train_tags = json.loads(line)
        for line in open('final_results_data/test_texts3.json', 'r'):
            val_texts = json.loads(line)
        for line in open('final_results_data/test_tags3.json', 'r'):
            val_tags = json.loads(line)
    elif k==3:
        for line in open('final_results_data/train_texts4.json', 'r'):
            train_texts = json.loads(line)
        for line in open('final_results_data/train_tags4.json', 'r'):
            train_tags = json.loads(line)
        for line in open('final_results_data/test_texts4.json', 'r'):
            val_texts = json.loads(line)
        for line in open('final_results_data/test_tags4.json', 'r'):
            val_tags = json.loads(line)
    elif k==4:
        for line in open('final_results_data/train_texts5.json', 'r'):
            train_texts = json.loads(line)
        for line in open('final_results_data/train_tags5.json', 'r'):
            train_tags = json.loads(line)
        for line in open('final_results_data/test_texts5.json', 'r'):
            val_texts = json.loads(line)
        for line in open('final_results_data/test_tags5.json', 'r'):
            val_tags = json.loads(line)
            
    train_tags, train_texts = fix_tag_problems(train_tags, train_texts)
    val_tags, val_texts = fix_tag_problems(val_tags, val_texts) 
     
    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    
    train_labels = encode_tags2(train_tags, train_encodings)
    val_labels = encode_tags2(val_tags, val_encodings)
    
    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")
    train_dataset = NERDataset(train_encodings, train_labels)
    val_dataset = NERDataset(val_encodings, val_labels)
    
    
    # BERT + Fine Tuning
      
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"]="5"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=len(unique_tags))
    
    model.to(device)
    
    training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=10,              # total number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=169,                # number of warmup steps for learning rate scheduler
            weight_decay=0.0042,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=500,
            evaluation_strategy="epoch",
            overwrite_output_dir = True,
            save_total_limit = 5,
            save_steps = 4000,
            learning_rate=6.917e-05,
        )
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )
    
    print("Training start: ")
    print(dt.datetime.now())
    t1 = time.time()
    trainer.train()
    result = trainer.evaluate()
    print(result)
    print("Time taken: "+str((time.time()-t1)/60)+" min")
    trainer.save_model(output_dir='./cur_model')
