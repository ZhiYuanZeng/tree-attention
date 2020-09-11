import torch
import dill
import os
import torchtext.data as data
from nltk.tree import Tree
from tree2matrix import tree2matrix, gen_random_tree
from transformers import BertTokenizer


class SST(data.Dataset):
    urls = ['http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip']
    dirname = 'trees'
    name = 'sst'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, index_field, subtrees=False,
                 fine_grained=False, **kwargs):
        """Create an SST dataset instance given a path and fields.

        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            subtrees: Whether to include sentiment-tagged subphrases
                in addition to complete examples. Default: False.
            fine_grained: Whether to use 5-class instead of 3-class
                labeling. Default: False.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field), ('index', index_field)]

        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[label]
        label_field.preprocessing = data.Pipeline(get_label_str)
        with open(os.path.expanduser(path)) as f:
            if subtrees:
                examples = []
                for i,line in enumerate(f):
                    exs=data.Example.fromtree(line, fields, True)
                    for e in exs:
                        setattr(e,'index',len(examples))
                        examples.append(e)
            else:
                examples = []
                for i,line in enumerate(f):
                    e=data.Example.fromtree(line, fields)
                    setattr(e,'index',i)
                    examples.append(e)
        super(SST, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, index_field, root='.data',
               train='train.txt', validation='dev.txt', test='test.txt',
               train_subtrees=False, **kwargs):
        """Create dataset objects for splits of the SST dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.txt'.
            train_subtrees: Whether to use all subtrees in the training set.
                Default: False.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), text_field, label_field, index_field, subtrees=train_subtrees,
            **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), text_field, label_field, index_field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), text_field, label_field, index_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

def process_sents(fine_grained=False, subtrees=False):
    def filter_neutral(examples): return [e for e in examples if e.label!='neutral']
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL =data.Field(sequential=False, unk_token=False)
    INDEX=data.Field(sequential=False,use_vocab=False)
    train,dev,test=SST.splits(TEXT,LABEL,INDEX, root='data/',fine_grained=fine_grained, train_subtrees=subtrees)
    if not fine_grained:
        train.examples=filter_neutral(train.examples)
        dev.examples=filter_neutral(dev.examples)
        test.examples=filter_neutral(test.examples)
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    print('saving dataset to data/')
    if not subtrees:
        torch.save(train.examples,'data/train.sst',pickle_module=dill)
    else:
        torch.save(train.examples,'data/train.all',pickle_module=dill)
    torch.save(dev.examples,'data/val.sst',pickle_module=dill)
    torch.save(test.examples,'data/test.sst',pickle_module=dill)
    if fine_grained:
        torch.save(train.fields,'data/fields5',pickle_module=dill) # fine_grained
    else:
        torch.save(train.fields,'data/fields2',pickle_module=dill) # binary

def trees2matrices_and_texts(path, subtrees=True, random_tree=None):
    matrices=[]
    texts=[]
    def get_subtrees(t): return [t for t in t.subtrees() if t.label()!='2']
    with open(os.path.expanduser(path)) as f:
        for line in f:
            t=Tree.fromstring(line)
            if random_tree is not None:
                t=gen_random_tree(random_tree, t.leaves())
                matrices.append(t)
                texts.append(t.leaves())
            elif subtrees:
                strees=get_subtrees(t)                
                matrices.extend([tree2matrix(t) for t in strees])
                strees=get_subtrees(t)
                texts.extend([t.leaves() for t in strees])
            else:
                matrices.append(t)
                texts.append(t.leaves())
            assert len(matrices)==len(texts)
    return matrices, texts

# for fine-tuning bert
def save_sst_to_tsv(fine_grained=False, sst_dir='data/sst/trees'):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL =data.Field(sequential=False, unk_token=False)
    INDEX=data.Field(sequential=False,use_vocab=False)
    train,val,test=SST.splits(TEXT,LABEL,INDEX, root='data/',fine_grained=fine_grained, train_subtrees=True)

    train.examples=[e for e in train.examples if e.label!='neutral']
    val.examples=[e for e in val.examples if e.label!='neutral']
    test.examples=[e for e in test.examples if e.label!='neutral']

    train_tsv_name='train.sst5.tsv' if fine_grained else 'train.tsv'
    val_tsv_name='val.sst5.tsv' if fine_grained else 'val.tsv'
    test_tsv_name='test.sst5.tsv' if fine_grained else 'test.tsv'
    tsv_dir=os.path.join('data/','fine-tune')
    
    if fine_grained:
        str2label= {'very negative':'0', 'negative':'1', 'neutral':'2',
                'positive':'3', 'very positive':'4'}
    else:
        str2label= {'negative':'0', 'positive':'1', }
        
    def save_to_tsv(examples, fname):
        with open(fname, 'w') as f:
            f.write('sentence\tlabel\n')
            for e in examples:
                t,l=e.text, e.label
                t=' '.join(t)
                l=str2label[l]
                f.write(f'{t}\t{l}\n') # tsv format
    save_to_tsv(train, os.path.join(tsv_dir,train_tsv_name))
    save_to_tsv(val, os.path.join(tsv_dir,val_tsv_name))
    save_to_tsv(test, os.path.join(tsv_dir,test_tsv_name))

# for fine-tuning bert
def process_trees_and_bpes(sst_dir='.data/sst/trees/', tokenizer:BertTokenizer=None, 
        save_dir='data/sst/fine-tune', save_bpe=True, **kwargs):
    train_matrices, train_texts=trees2matrices_and_texts(os.path.join(sst_dir,'train.txt'), **kwargs)
    torch.save(train_matrices, os.path.join(save_dir, 'trees'))
    if save_bpe:
        all_bpe_indices=[]
        for text in train_texts: # text is a list of tokens
            tokens=tokenizer.tokenize(' '.join(text))
            bpe_indices=[]
            for i,t in enumerate(tokens):
                if '##' in t:
                    bpe_indices.append(i)
            all_bpe_indices.append(bpe_indices)
            assert len(text)==len(tokens)-len(bpe_indices)
        torch.save(all_bpe_indices, os.path.join(save_dir, 'bpe'))


if __name__ == "__main__":
    # from process_data import SST
    # process_sents(fine_grained=False, subtrees=True)

    # save_sst_to_tsv(fine_grained=False)

    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', split_puntc=False)
    process_trees_and_bpes(tokenizer=tokenizer)