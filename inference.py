# Implementation via huggingface transformers
# setup 2

import json
import numpy as np
from seqeval.metrics import f1_score, classification_report
from seqeval.metrics.v1 import precision_recall_fscore_support
from seqeval.scheme import BILOU
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from transformers import DistilBertTokenizerFast, DistilBertTokenizer, DistilBertForTokenClassification, Trainer, TrainingArguments, BertTokenizerFast, BertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, AutoModel
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import time
import os
from datasets import load_dataset, load_metric
from typing import Dict

# load data texts and labels
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
        
        #labelling the special tokens CLS and SEP:
        doc_enc_labels[0] = tag2id['O']
        last_index = doc_input_ids.index(102)
        doc_enc_labels[last_index] = tag2id['O']
        
        # adjusting label positions to the created subtokens
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
    
    result = classification_report(y_true=l, y_pred=p, scheme=BILOU, mode='strict')
    print(result)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=l, y_pred=p, average="weighted", scheme=BILOU)
    
    return {
            'precision': precision,
            'recall': recall,
            'f1': f1,}

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
    return tags
    

# main:
    
tags = fix_tag_problems(tags, texts) 

# 80/20 train-test split of the data
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2, random_state = 51)
    
# integer representation of the labels with positional tagging
unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {'B-NEG_FINDINGS': 0,
 'I-FAMILY': 1,
 'B-ETHNICITY': 2,
 'L-FAMILY': 3,
 'U-MEDICATION': 4,
 'B-SYMPTOMS': 5,
 'U-DIAGNOSIS': 6,
 'L-GENDER': 7,
 'O': 8,
 'I-GENETICS': 9,
 'I-GENDER': 10,
 'B-MEDICATION': 11,
 'I-BIOCHEMICAL': 12,
 'I-NEG_FINDINGS': 13,
 'B-PARACLINICAL': 14,
 'B-BIOCHEMICAL': 15,
 'B-GENDER': 16,
 'B-DIAGNOSIS': 17,
 'L-AGE': 18,
 'L-PARACLINICAL': 19,
 'U-ETHNICITY': 20,
 'L-SYMPTOMS': 21,
 'L-ETHNICITY': 22,
 'I-DIAGNOSIS': 23,
 'I-AGE': 24,
 'U-PARACLINICAL': 25,
 'U-BIOCHEMICAL': 26,
 'B-GENETICS': 27,
 'U-FAMILY': 28,
 'I-ETHNICITY': 29,
 'L-GENETICS': 30,
 'U-NEG_FINDINGS': 31,
 'B-AGE': 32,
 'I-SYMPTOMS': 33,
 'U-GENETICS': 34,
 'B-FAMILY': 35,
 'L-BIOCHEMICAL': 36,
 'U-GENDER': 37,
 'L-DIAGNOSIS': 38,
 'L-MEDICATION': 39,
 'I-PARACLINICAL': 40,
 'L-NEG_FINDINGS': 41,
 'U-AGE': 42,
 'U-SYMPTOMS': 43,
 'I-MEDICATION': 44}
id2tag = {id: tag for tag, id in tag2id.items()}

# encoding the data
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

# matching the tags to the corresponding subtokens
train_labels = encode_tags2(train_tags, train_encodings)
val_labels = encode_tags2(val_tags, val_encodings)

# removes the offset mappings and defines the training dataset and the validation dataset
train_encodings.pop("offset_mapping")
val_encodings.pop("offset_mapping")
train_dataset = NERDataset(train_encodings, train_labels)
val_dataset = NERDataset(val_encodings, val_labels)


# BERT + Fine Tuning

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training BioBERT v1.1 (BERT_BASE initially)
#model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=len(unique_tags))
# using the current model
model = AutoModelForTokenClassification.from_pretrained('./cur_model')

model.to(device)

training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=169,                # number of warmup steps for learning rate scheduler
        weight_decay=0.0042,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        evaluation_strategy="epoch",
        overwrite_output_dir = True,
        save_total_limit = 5,
        save_steps = 500,
        learning_rate=6.917e-05,
    )

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
)

print("Evaluation start: ")
print(dt.datetime.now())
t1 = time.time()
result = trainer.evaluate()
print(result)
print("Time taken: "+str((time.time()-t1)/60)+" min")


sequence = "Oculomotor apraxia may be idiopathic or a symptom of a variety of diseases. In Gaucher disease, oculomotor deficit is characterized by a failure of volitional horizontal gaze with preservation of vertical movements. We present 2 sisters, 6 1/2 and 5 1/2 years of age, in whom the presenting sign was oculomotor apraxia. Oculomotor apraxia has not been previously reported as the presenting manifestation of Gaucher disease."
sequence2 = "Gaucher's disease has been associated with plasma cell dyscrasias. A patient had Gaucher's disease, nephrotic syndrome, and systemic amyloidosis. Plasmacytosis in the bone marrow, the presence of light chains in the urine and renal glomeruli, and the finding of low circulating immunoglobulin levels suggest that the amyloid in this patient is related to a plasma cell dyscrasia."
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")
outputs = model(inputs).logits
predictions = torch.argmax(outputs, dim=2)
print([(token, id2tag[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])
