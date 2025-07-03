"""
gradient -> (f(x+h) - f(x) / h) 

All the operations are decomposed into sums and adds and exp (everything is fine, as long as I know how to compute the local derivative)

Each value is a node in the tree    
Each value stores the tree of past operations required to reach that value  

Grad indicates how much the loss function is impacted by the weight inside that value (dL/d_node) 

ChainRule -> dL / dc = dL/dd * dd/dc
So when I have a "+" node, its like multipling by one the gradient during back prop.
es. e = a*b -> de/da = b

Topological Sort allows to go only to the right 

"""

import math
import random

class Value: 

    def __init__(self, data, children=(), op=''):
        self.data = data
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None
        self.grad = 0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad += 1*out.grad  # add chainrule
            other.grad += 1*out.grad # add  chainrule
        out._backward = backward
        return out
    
    def __radd__(self, other):
        return self + other
   
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')
        def backward():
            self.grad += 1*out.grad  # sub chainrule for first operand
            other.grad += -1*out.grad # sub chainrule for second operand
        out._backward = backward
        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self
   
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad += other.data*out.grad # mul chain rule   
            other.grad += self.data*out.grad # mul chain rule
        out._backward = backward
        return out
    
    def __rmul__(self, other):
        return self * other
         
    def __pow__(self, other):
        out = Value(self.data**other, (self,), f'**{other}')
        def backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = backward
        return out
         
    def tanh(self):
        x = self.data
        res = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        out =  Value(res, (self,), 'tanh')
        def backward():
            self.grad += (1 - res**2) * out.grad
        out._backward = backward    
        return out  

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def backward(self): 
        ## TOPO SORT    
        topo = []   
        visited = set() 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)  
        build_topo(self)
        self.grad = 1 # inizialize
        # AUTOMATE BACKPROP
        for node in reversed(topo): # from right to left
            node._backward()



class Neuron: 
    def __init__(self, n_in):
        self.weight = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.bias= Value(random.uniform(-1,1))

    def __call__(self,x): 
        act = sum(wi*xi for wi,xi in zip(self.weight, x)) + self.bias
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.weight + [self.bias]
    

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        params = []
        for n in self.neurons:
            ps = n.parameters() 
            params.extend(ps)
        return params
    
class MLP:
    def __init__(self, n_in, n_out):
        size = [n_in] + n_out
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_out))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # output of each layer becomes input of the next one
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params

    def zero_grad(self): # reset gradients otherwise they sum up
        for p in self.parameters():
            p.grad = 0


################ START EXAMPLE ##################

n = MLP(3, [4,4,1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
y_truth = [-1.0, 1.0, -1.0, 1.0]
starting_preds = [n(x) for x in xs]

for i in range(100):
   
    # forward pass
    y_pred = [n(x) for x in xs]
    loss = sum((ytarget - ypredicted)**2 for ytarget, ypredicted in zip(y_truth, y_pred))
    print(i, loss.data)

    # zero gradients
    n.zero_grad()

    # backward pass
    loss.backward()

    # update weights
    for p in n.parameters():
        p.data += -0.05*p.grad

print("")
print("")
print("#"*50)
print("STARTING PREDS:",  [round(y.data,2) for y in starting_preds])
print("FINAL PREDS :",  [round(y.data,2) for y in y_pred])
print("TRUTH VALUES:",  y_truth)
print("#"*50)
print("")
print("")