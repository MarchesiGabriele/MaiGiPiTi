"""
gradient -> (f(x+h) - f(x) / h) 

All the operations are decomposed into sums and adds and exp (everything is fine, as long as I know how to compute the local derivative)

Each value is a node in the tree    
Each value stores the tree of past operations required to reach that value  

Grad indicates how much the loss function is impacted by the weight inside that value (dL/d_node) 

ChainRule -> dL / dc = dL/dd * dd/dc
So when I have a "+" node, its like multipling by one the gradient during back prop.
es. e = a*b -> de/da = b

Topological Sort allows to go only to le rightn 

"""

import math

class Value: 

    def __init__(self, data, children=(), op=''):
        self.data = data
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None
        self.grad = 0

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad = 1*out.grad  # add chainrule
            other.grad = 1*out.grad # add  chainrule
        out._backward = backward
        return out
   
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad = other.data*out.grad # mul chain rule   
            other.grad = self.data*out.grad # mul chain rule
        out._backward = backward
        return out
         
    def tanh(self):
        x = self.data
        res = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        out =  Value(res, (self), 'tanh ')
        def backward():
            self.grad = 1 - res**2 * out.grad
        out._backward = backward    
        return out  

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
        for node in reversed(topo):
            node._backward()


# o.grad = 1    # last node gradient is always 1
# o._backward()