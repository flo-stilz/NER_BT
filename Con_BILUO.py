# DATA Conversion to BILUO tags

import spacy
import nltk
import json

def preprocess_data():
    raw_data = []
    for line in open('data.jsonl', 'r'):
        raw_data.append(json.loads(line))
    
    raw_data = raw_data
    data = []
    for abstract in raw_data:
        token_s = []
        token_e = []
        abstract_size = 0
        label = []
        sentence = []
        sentences = nltk.tokenize.sent_tokenize(abstract['text'])
    
        for span in abstract['spans']:
            token_s.append(span['start'])
            token_e.append(span['end'])
            label.append(span['label'])
            
        label_pos = 0
        s_count = 0
        for sentence in sentences:
            tags = []
            l = abstract_size + len(sentence)+s_count
            
            while len(label)>label_pos and token_e[label_pos]<=l:
                tags.append((token_s[label_pos]-abstract_size-s_count, token_e[label_pos]-abstract_size-s_count, label[label_pos]))
                label_pos+=1
            
            data.append((sentence, {'entities': tags}))
            abstract_size = abstract_size + len(sentence)
            s_count+=1
    return data

data = preprocess_data()
nlp = spacy.blank('en')
texts = []
tags = []
for text, annotation in data:
    doc = nlp(text)
    entities = annotation['entities']
    tags.append(spacy.gold.biluo_tags_from_offsets(doc,entities))
    subtext = []
    for token in doc:
        subtext.append(str(token))
    texts.append(subtext)

with open('tags.json', 'w') as outfile:
    json.dump(tags, outfile)
with open('texts.json', 'w') as outfile:
    json.dump(texts, outfile)