import torch 
import numpy as np

names = [name.replace('"', "").replace(',', "").lower() for name in open("animales.txt", "r").read().splitlines()]

chars = sorted(list(set("".join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)} 
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} 

# CREO DATASET
xs,ys = [],[]

N = torch.zeros((len(chars)+1,len(chars)+1), dtype=torch.int32)
for name in names[:1]:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): 
        N[stoi[ch1],stoi[ch2]] += 1 
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

x = torch.tensor(xs)
y = torch.tensor(ys)


import torch.nn.functional as F

xenc = F.one_hot(x, num_classes=len(chars)).float()

# len(chars) è il numero di input, 
# inizializzo i pesi del primo layer (questa dimensione di matrice serve moltiplicare tutti gli input per tutti i neuroni\)
W = torch.randn(len(chars),len(chars), requires_grad=True) 

a =  xenc @ W


