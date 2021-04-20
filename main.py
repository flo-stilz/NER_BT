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
'''
raw_data = []
for line in open('ner_gaucher_GAEDSMPBGNF.jsonl', 'r'):
    raw_data.append(json.loads(line))
'''
for line in open('texts.json', 'r'):
    texts = json.loads(line)
for line in open('tags.json', 'r'):
    tags = json.loads(line)

def preprocess_data(raw_data):
    texts = []
    tags = []
    
    for abstract in raw_data:
        token_s = []
        token_e = []
        label = []
        read_ab_size = 0
        s_pos = 0
        sentence = []
        label_sub = []
        s = nltk.tokenize.sent_tokenize(abstract['text'])
    
        for span in abstract['spans']:
            token_s.append(span['start'])
            token_e.append(span['end'])
            label.append(span['label'])
            
        label_pos = 0
        for token in abstract['tokens']:
            
            sentence.append(token['text'])
            
            if (token['start'] >= token_s[label_pos] and token['end'] <= token_e[label_pos]):
                
                label_sub.append(label[label_pos])
                
                if (token['end'] == token_e[label_pos]) and (len(token_e)-1!=label_pos):
                   label_pos+=1
               
            else:
                label_sub.append('O')
             
            
            if (s_pos<len(s) and token['end']-1 == len(s[s_pos])-1 + read_ab_size):
                if ((len(abstract['text']) > read_ab_size + len(s[s_pos])) and abstract['text'][read_ab_size+len(s[s_pos])] == ' '):
                    read_ab_size += len(s[s_pos])+1
                else:
                    read_ab_size += len(s[s_pos])
                s_pos += 1
                tags.append(label_sub)
                texts.append(sentence)
                label_sub = []
                sentence = []
                
    return texts, tags
    
def augment_training_data(texts, tags, limit):
    te = [[]]
    ta = [[]]
    for i in range(len(texts)):
        a = 0
        for j in range(len(texts[i])):
            if j>0 and j<len(tags[i]) and tags[i][j] == 'O' and a<limit:
                r = random.randint(0,10)
                if r<2:
                    te.append(copy.deepcopy(texts[i]))
                    ta.append(copy.deepcopy(tags[i]))
                    
                    del te[-1][j]
                    del ta[-1][j]
                    a+=1
                for k in range(len(texts[i])-(j+1)):
                    if j+k<len(tags[i]) and tags[i][j+k] == 'O' and a<limit:
                        r = random.randint(0,20)
                        if r<1:
                            te.append(copy.deepcopy(texts[i]))
                            ta.append(copy.deepcopy(tags[i]))
                            res = texts[i][j]
                            te[-1][j] = te[-1][j+k]
                            te[-1][j+k] = res
                            a+=1
    del te[0]
    del ta[0]
    te = texts + te
    ta = tags + ta
    return te, ta

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

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
    #print(dt.datetime.now())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=l, y_pred=p, average="weighted", scheme=BILOU)
    
    return {
            'precision': precision,
            'recall': recall,
            'f1': f1,}
    
def model_init():
    return AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=len(unique_tags), return_dict=True)

def fix_tags(tags):
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            if tags[i][j]=='-':
                tags[i][j]='O'
    return tags
    
tags = fix_tags(tags) 

CV = model_selection.KFold(5, shuffle=True)
      
for (k, (train_index, test_index)) in enumerate(CV.split(texts,tags)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,5))
#train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2, random_state = 51)

#train_texts, train_tags = augment_training_data(train_texts, train_tags, 5)
# only for hyperparameter testing!!!
    '''
    val_texts = train_texts[300:400] 
    val_tags = train_tags[300:400]
    train_texts = train_texts[:100]
    train_tags = train_tags[:100]
    '''
    
    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    train_texts = [texts[i] for i in train_index]
    train_tags = [tags[i] for i in train_index]
    val_texts = [texts[i] for i in test_index]
    val_tags = [tags[i] for i in test_index]
    
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    
    train_labels = encode_tags2(train_tags, train_encodings)
    val_labels = encode_tags2(val_tags, val_encodings)
    
    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    train_dataset = NERDataset(train_encodings, train_labels)
    val_dataset = NERDataset(val_encodings, val_labels)
    
    
    # BERT + Fine Tuning
      
    # Additional parameter example:
    # model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='bert-base-uncased', num_labels=2, output_attentions = False, output_hidden_states = False, attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=len(unique_tags), hidden_dropout_prob=0.5)
    #model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))
    #model = BertForTokenClassification.from_pretrained('bert-large-cased', num_labels=len(unique_tags))
    #model = DistilBertForTokenClassification.from_pretrained('./cur_model')
    
    #model.to(device)
    
    training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=9,              # total number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=5,                # number of warmup steps for learning rate scheduler
            weight_decay=0.089,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=500,
            evaluation_strategy="epoch",
            overwrite_output_dir = True,
            save_total_limit = 5,
            save_steps = 4000,
            learning_rate=8.867e-05
        )
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )
    '''
    def hp_space_optuna(trial) -> Dict[str, float]:
        from transformers.integrations import is_optuna_available
    
        assert is_optuna_available(), "This function needs Optuna installed: `pip install optuna`"
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 10),
            "seed": trial.suggest_int("seed", 1, 40),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 0.001, 1, log=True),
            "warmup_steps": trial.suggest_int("warmup_steps", 1, 500),
        }
    
    best_trials = trainer.hyperparameter_search(
        direction="maximize", 
        backend="optuna",
        n_trials = 300,
        hp_space = hp_space_optuna
    )
    
    print(best_trials)
    '''
    print("Training start: ")
    print(dt.datetime.now())
    t1 = time.time()
    trainer.train()
    result = trainer.evaluate()
    print(result)
    print("Time taken: "+str((time.time()-t1)/60)+" min")
    trainer.save_model(output_dir='./cur_model')
    #model = DistilBertForTokenClassification.from_pretrained('./cur_model')
