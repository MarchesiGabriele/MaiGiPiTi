import torch

names = [name.replace('"', "").replace(',', "") for name in open("animales.txt", "r").read().splitlines()]


chars = sorted(list(set("".join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}

N = torch.zeros((len(chars)+1,len(chars)+1), dtype=torch.int32)

for name in names[:5]:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1],stoi[ch2]] += 1


print(N)











