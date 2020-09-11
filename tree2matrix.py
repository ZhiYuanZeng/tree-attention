from nltk.tree import Tree
from nltk.corpus import ptb
import numpy as np

def tree2matrix(t):
    all_spans=set()
    def len_of_tree(beg_idx,t):
        if isinstance(t,str) or isinstance(t,int): return 1
        L=0
        for subt in t:
            subl=len_of_tree(beg_idx,subt)
            all_spans.add((beg_idx,beg_idx+subl-1))
            beg_idx+=subl
            L+=subl
        return L
    L=len_of_tree(0,t)
    all_spans.add((0,L-1))
    matrix=np.zeros([L,L])
    for span in all_spans:
        beg_idx,end_idx=span
        matrix[beg_idx:end_idx+1,beg_idx:end_idx+1]+=1
    matrix=normalize(matrix)
    # print(matrix.sum(axis=1))
    assert np.all(matrix.sum(axis=1)-1<1e-6)
    # matrix=matrix/matrix.sum()*len(matrix)
    # assert matrix.sum()-len(matrix)< 1e-6
    return matrix

def matrix2tree(matrix,leaves)->Tree:
    def parse(beg_idx,end_idx):
        loc_matrix=matrix[beg_idx:end_idx+1,beg_idx:end_idx+1]
        if beg_idx==end_idx: return Tree('na',[leaves[beg_idx]])
        sub_trees=[]
        L=len(loc_matrix)
        all_sum,all_area=loc_matrix.sum(),L**2
        split_scores=[]
        split_indices=[-1]
        for i in range(L-1):
            left_up=loc_matrix[0:i+1,0:i+1]
            right_down=loc_matrix[i+1:L,i+1:L]
            left_up_sum,right_down_sum=left_up.sum(),right_down.sum()
            left_up_area,right_down_area=(i+1)**2,(L-(i+1))**2
            other_sum=all_sum-left_up_sum-right_down_sum
            other_area=all_area-left_up_area-right_down_area
            # score=(left_up_sum+right_down_sum)/(left_up_area+right_down_area)-other_sum/other_area
            score=all_sum/all_area-other_sum/other_area
            split_scores.append(score)
        max_score=max(split_scores)
        for j,s in enumerate(split_scores):
            if s==max_score:
                split_indices.append(j)
        split_indices.append(L-1)
        for j,idx in enumerate(split_indices):
            if j==len(split_indices)-1: break
            next_idx=split_indices[j+1]
            assert next_idx>=idx
            subt=parse(beg_idx+idx+1,beg_idx+next_idx)
            sub_trees.append(subt)
        t=Tree('na',sub_trees)
        return t
    return parse(0,len(leaves)-1)

def normalize(z):
    return z/z.sum(axis=-1,keepdims=True)

def softmax(z):
    return np.exp(z)/np.exp(z).sum(axis=-1,keepdims=True)

# %%
def gen_random_tree(tree_type, leaves):
    leaves=[str(w) for w in leaves]
    if tree_type=='right':
        # right branch tree
        root=Tree('na',[leaves[-1],])
        for i,w in enumerate(reversed(leaves)):
            if i==0: continue
            sub_trees=[w, root]
            root=Tree('na', sub_trees)
    elif tree_type=='balance':
        raise NotImplementedError
    return root
# %%
# if __name__ == "__main__":
#     with open('/data/zyzeng/datasets/treebank/en/WSJ/22/WSJ_2202.MRG','r') as f:
#         # check
#         with open('model/demo.mrg','r') as f:
#             s=f.read()
#             t=Tree.fromstring(s)
#             tree2matrix(t)
#             exit()
#         fileids=ptb.fileids()
#         val_ids=[]
#         for i in fileids:
#             if i.startswith('WSJ/22'):
#                 val_ids.append(i)
#         trees=ptb.parsed_sents(val_ids)
#         for t in trees:
#             matrix=tree2matrix(t)
#             # matrix=softmax(matrix)
#             _tree=matrix2tree(matrix,t.leaves())
#             _matrix=tree2matrix(_tree)
#             # _matrix=softmax(_matrix)
#             assert np.all(_matrix==matrix)