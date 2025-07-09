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
for name in names:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): 
        N[stoi[ch1],stoi[ch2]] += 1 
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

x = torch.tensor(xs)
y = torch.tensor(ys)
l = len(x) # numer of inputs

import torch.nn.functional as F

# initialize weigths to random
W = torch.randn(28,28, requires_grad=True)  # len(chars) = num of inputs

for k in range(10000):
    ####################### FORWARD PASS ##################################
    # hot encode the inputs
    xenc = F.one_hot(x, num_classes=28).float()

    # feed the inputs into the net
    logits =  xenc @ W # log counts
    counts = logits.exp() # counters of next char 

    # obtained probs of next char (given the input one)
    prob = counts / counts.sum(1, keepdim=True) # [n_inputs x 28]

    # Compute loss 
    my_probs = prob[torch.arange(l), y] # get from the net the probability of the next char given the input (higher is better, since i'm using the input and the truth label)
    loss = -my_probs.log().mean() # compute negative log like

    print("NLL: ", loss.item()) 
        
    ####################### BACKWARD PASS ##################################
    W.grad = None # reset gradients
    loss.backward() 

    
    ####################### UPDATE ##################################
    W.data += -0.5*W.grad









