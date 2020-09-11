from torch import nn
import torch
import dill
import functools
from torchtext.data import Dataset
from torchtext.data import BucketIterator
from tqdm import tqdm
from torch.nn.utils import clip_grad_value_
from models.lstm_classifiter import LstmModel
from models.transformer_classifiter import TransformerModel
from process_data import SST
from tree2matrix import matrix2tree, tree2matrix
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')
torch_load=functools.partial(torch.load, pickle_module=dill)

device=torch.device(1)

def train(model, train_dataset, val_dataset=None, trees=None, EM=False,
         lambda_=0.4,epoch=50, print_dec=400):
    def get_acc(preds, targets): return sum([1 if p[t]>0.5 else 0 for p,t in zip(preds,targets)])
    optimizer=torch.optim.Adam(model.parameters())
    crit=nn.CrossEntropyLoss()
    step, train_loss, train_acc=0, 0., 0.
    for e in range(epoch):
        print('epoch: {}----------------------------------'.format(e))
        model.train()
        for i,batch in enumerate(tqdm(train_dataset)):
            (inputs, lens),label, indices=batch.text, batch.label, batch.index
            inputs, label=inputs.to(device), label.to(device)
            bacth_size=len(inputs)
            if isinstance(model, LstmModel): 
                outs=model(inputs)
            else: # transformer
                outs, attns=model(inputs) # attns: list of tensor(bsz, head, e, e)
            loss=crit(outs, label-1)

            if trees is not None and lambda_!=0: 
                aux_loss=0
                attns=torch.stack(attns, dim=0) # layer, bsz, head, e, e 
                attns=attns.transpose(0,1) # bsz, layer, head, e, e 
                for i,a in enumerate(attns):
                    t=trees[indices[i]]
                    a=a[:,:1]
                    l=lens[i] # length of inputs is larger than real length, because of pad
                    a=a[:, :, :l, :l]
                    assert a.shape[-2:]==t.shape
                    aux_loss+=torch.norm(a-t, p=2, dim=(-2,-1)).mean()
                aux_loss=lambda_*aux_loss/bacth_size
                loss+=aux_loss

            loss.backward()
            train_loss+=loss.item()
            train_acc+=get_acc(outs, label-1)/bacth_size

            clip_grad_value_(model.parameters(), 1.) # grad clip
            optimizer.step()
            model.zero_grad()
            step+=1
            if step!=0 and step%print_dec==0:
                print(f'loss in batch {i}:{train_loss/print_dec} acc:{train_acc/print_dec}')
                train_loss=0
                train_acc=0
        # E step, update tree
        if lambda_!=0 and trees is not None and EM:
            trees=attention2tree(model, train_dataset, trees)
        # eval
        model.eval()
        if val_dataset is None: continue
        sample_num=0
        acc=0
        with torch.no_grad():
            for i,batch in enumerate(val_dataset):
                (inputs, lens),label=batch.text, batch.label
                inputs, label=inputs.to(device), label.to(device)
                if isinstance(model, LstmModel): 
                    outs=model(inputs)
                else: # transformer
                    outs, attns=model(inputs) # attns: list of tensor(bsz, head, e, e)
                sample_num+=len(outs)
                acc+=get_acc(outs, label-1)
            print('-------------val acc: ', acc/sample_num)
            writer.add_scalar('val_acc', acc/sample_num, step)
            
    return model

def attention2tree(model, data, trees):
    model.eval()
    with torch.no_grad():
        for batch in data:
            (inputs, lens),_, indices=batch.text, batch.label, batch.index
            inputs =inputs.to(device)
            _, attns=model(inputs)
            attns=torch.stack(attns, dim=0) # layer, bsz, head, e, e 
            attns=attns.transpose(0,1) # bsz, layer, head, e, e 
            for i,a in enumerate(attns):
                a=a[:,:1]
                l=lens[i] # length of inputs is larger than real length, because of pad
                a=a[:, :, :l, :l]
                tree=matrix2tree(a.mean(dim=(0,1)).cpu().numpy(), list(range(l)))
                matrix=tree2matrix(tree)
                assert trees[indices[i]].shape==matrix.shape
                trees[indices[i]]=torch.from_numpy(matrix).to(device)
    return trees


if __name__ == "__main__":
    # read sents
    fields=torch.load('data/fields2')
    train_examples=torch.load('data/train.all')
    val_examples=torch.load('data/val.sst')
    print('length of examples: ', len(train_examples))
    # read trees
    trees=torch.load('data/trees/trees.all')
    trees=[torch.from_numpy(t).to(device) for t in trees]
    assert all([len(trees[e.index])==len(e.text) for e in train_examples])

    train_data=Dataset(train_examples, fields)
    val_data=Dataset(val_examples, fields)
    train_iter,val_iter=BucketIterator.splits(
        (train_data, val_data),
        batch_sizes=(128,128),
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=True
    )

    # model=LstmModel(num_embed=len(fields['text'].vocab), embed_dim=64, out_dim=2, lstm_layer_num=2)
    model=TransformerModel(
        len(fields['text'].vocab),
        d_model=64,
        nhead=4,
        dim_feedforward=256,
        num_layers=2,
        out_dim=2
    )
    model.to(device)
    train(model, train_iter, val_iter, trees, print_dec=400, lambda_=0.5, EM=False)