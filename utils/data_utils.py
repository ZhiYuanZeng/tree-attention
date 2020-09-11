import torch

def remove_bpe_from_attention(bpe_ids, attention):
    """
    remove bpe id of one sentence
    args
        bpe_ids: List
        attention: Tensor (layer, head, seq,seq)
    """
    seq_len=attention.shape[-1]
    device=attention.device
    no_bpe_attention=attention.clone()
    non_bpe_ids=[]
    cnt=1
    if isinstance(bpe_ids,tuple):  bpe_ids=bpe_ids[0]
    for i in reversed(range(seq_len)):
        if i not in bpe_ids:
            cnt=1
            non_bpe_ids.append(i)
        else:
            cnt+=1
            no_bpe_attention[:,:,i-1]=no_bpe_attention[:,:,i-1]/cnt+no_bpe_attention[:,:,i]*(cnt-1)/cnt
            no_bpe_attention[:,:,:,i-1]=(no_bpe_attention[:,:,:,i]+no_bpe_attention[:,:,:,i-1])
    non_bpe_ids.reverse()
    assert len(non_bpe_ids)==(seq_len-len(bpe_ids))
    no_bpe_attention=torch.index_select(no_bpe_attention,dim=-2,index=torch.tensor(non_bpe_ids).to(device))
    no_bpe_attention=torch.index_select(no_bpe_attention,dim=-1,index=torch.tensor(non_bpe_ids).to(device))
    return no_bpe_attention