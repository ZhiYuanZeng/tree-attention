import torch
import dill
import os
import torchtext.data as data
from torchtext.datasets import IMDB
from nltk.tree import Tree
from tree2matrix import tree2matrix, gen_random_tree
from transformers import BertTokenizer

def process_sents():
    def insert_index(dataset:data.Dataset):
        examples=dataset.examples
        fields=dataset.fields
        for i,e in enumerate(examples):
            setattr(e, 'index', i)
        fields['index']=data.Field(sequential=False, use_vocab=False)
        dataset.examples=examples
        dataset.fields=fields
        return dataset

    text=data.Field(lower=True, include_lengths=True)
    label=data.Field(sequential=False, is_target=True, use_vocab=False)
    train_data, test_data=IMDB.splits(text, label)
    train_data=insert_index(train_data)
    test_data=insert_index(test_data)

    # save data
    torch.save(train_data.examples, 'data/imdb/train.data')
    torch.save(test_data.examples, 'data/imdb/test.data')
    torch.save(train_data.fields, 'data/imdb/fields')

def save_imdb_to_tsv():
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL =data.Field(sequential=False, unk_token=False)
    train,val=IMDB.splits(TEXT,LABEL)
    
    str2label= {'negative':'0', 'positive':'1', }
    test=val.examples
    dev=train.examples[-len(test):]
    train=train.examples[:-len(test)]
    def save_to_tsv(examples, fname):
        with open(fname, 'w') as f:
            f.write('sentence\tlabel\n')
            for e in examples:
                t,l=e.text, e.label
                t=' '.join(t)
                l=str2label[l]
                f.write(f'{t}\t{l}\n') # tsv format
    tsv_dir='data/imdb/fine-tune'
    save_to_tsv(train, os.path.join(tsv_dir,'train.tsv'))
    save_to_tsv(dev, os.path.join(tsv_dir,'dev.tsv'))
    save_to_tsv(test, os.path.join(tsv_dir,'test.tsv'))


if __name__ == "__main__":
    save_imdb_to_tsv()