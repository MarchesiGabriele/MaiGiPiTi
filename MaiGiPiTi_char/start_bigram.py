import torch

names = [name.replace('"', "").replace(',', "").lower() for name in open("animales.txt", "r").read().splitlines()]

## Get all the possible characheters
# NB: tra i caratteri c'è anche lo spazio vuoto
chars = sorted(list(set("".join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)} # associo a ciascun carattere un numero
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()} # viceversa

N = torch.zeros((len(chars)+1,len(chars)+1), dtype=torch.int32)

## Calcolo quante volte si presenta ogni coppia di lettere
for name in names:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # creo le combinazioni tra lettere successive per ciascuna parola
        N[stoi[ch1],stoi[ch2]] += 1 # ogni volta che ho una coppia di lettere la salvo

# normalizzo ciascuna riga
p = (N+1).float() / N.float().sum(dim=1, keepdim=True)

newnames = []
idx = 0
gen = torch.Generator().manual_seed(2839232938298)
for seed in range(10):
    name = "" 
    while True: 
        idx = torch.multinomial(p[idx], 1, replacement=True, generator=gen).item() # estraggo token successivo
        name += itos[idx]
        if idx == 0:
            break
    newnames.append(name)
print(newnames)


##### COMPUTING LOG-LIKELIHOOOD ###########
n = 0
loglike = 0.0 
for name in names[:3]:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # creo le combinazioni tra lettere successive per ciascuna parola
        prob = p[stoi[ch1],stoi[ch2]] # uso le prob. imparate precedentemente
        logprob = torch.log(prob)
        n += 1
        print(f"{ch1}, {ch2}, : {prob:.4f}, {logprob:.4f}")
        loglike += logprob

# make it negative
print(f"NEGATIVE LOGLIKE: ", f"{-loglike}")
print(f"AVERAGE LOGLIKE: ", f"{-loglike/n}")











