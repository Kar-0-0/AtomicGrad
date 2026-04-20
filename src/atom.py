from math import log  # Hopefully we can remove this at some point

class Tensor:
    def __init__(self, data, _children=()):
        self.data = data
        self.shape = self.get_shape(data)
        self.children = set(_children)
        self.grad = None
        self._backward = lambda: None


    def __matmul__(self, other):
        if not isinstance(self, Tensor) or not isinstance(other, Tensor):
            raise TypeError("Both values must be Tensors")

        # BxMxN @ BxNxK = BxMxK
        assert self.shape[-1] == other.shape[-2]
        if len(other.shape) == 3:
            assert self.shape[0] == other.shape[0] 
            
        if len(self.shape) == 2:
            M, K = self.shape[0], other.shape[-1]
            out = zeros(M, K)
        else:
            B, M, K = self.shape[0], self.shape[1], other.shape[-1]
            out = zeros(B, M, K)

        if len(self.shape) == 3:
            for b in range(B):
                mat1 = self.data[b]
                if len(other.shape) == 3:
                    mat2 = other.data[b]
                else:
                    mat2 = other.data
                
                for i in range(M):
                    for j in range(K):
                        acc = 0
                        for k in range(self.shape[-1]):
                            acc += mat1[i][k] * mat2[k][j]

                        out[b][i][j] = acc
        
        elif len(self.shape) == 2 and len(other.shape) == 2:
            for i in range(M):
                for j in range(K):
                    acc = 0
                    for k in range(self.shape[-1]):
                        acc += self.data[i][k] * other.data[k][j]
                
                    out[i][j] = acc

        def _backward():
            self.grad += out.grad.__matmul__(other.transpose()) 
            other.grad += self.transpose().__matmul__(out.grad)  

        out._backward = _backward
        out.children = {self, other}

        return out

    def __add__(self, other):
        out = zeros(*self.shape)
        if isinstance(other, (int, float)):
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    if len(self.shape) == 3:
                        for j in range(self.shape[2]):
                            out[b][i][j] = self.data[b][i][j] + other
                    else:
                            out[b][i] = self.data[b][i] + other

            
        else:
            assert isinstance(self, Tensor) and isinstance(other, Tensor) and self.shape == other.shape
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    if len(self.shape) == 3:
                        for j in range(self.shape[2]):
                            out[b][i][j] = self.data[b][i][j] + other[b][i][j]
                    else:
                            out[b][i] = self.data[b][i] + other[b][i]

        def _backward():
            self.grad += out.grad       
            if isinstance(other, Tensor):
                other.grad += out.grad
        
        out.children = {self, other} if isinstance(other, Tensor) else {self}
        out._backward = _backward

        return out

    def __mul__(self, other):
        if isinstance(self, Tensor) and isinstance(other, Tensor):
            assert self.shape == other.shape

            out = zeros(*self.shape)
            if len(self.shape) == 3:
                for b in range(self.shape[0]):
                    for i in range(self.shape[1]):
                        for j in range(self.shape[2]):
                            out[b][i][j] = self.data[b][i][j] * other.data[b][i][j]
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        out[i][j] = self.data[i][j] * other.data[i][j]
            
            return out

    def __truediv__(self, other):
        out = zeros(*self.shape)
        if len(self.shape) == 3:
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        if isinstance(other, Tensor):
                            out[b][i][j] = self.data[b][i][j] / other[b][i][j]
                        elif isinstance(other, (int, float)):
                            out[i][j] = self.data[i][j] / other

        else:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if isinstance(other, Tensor):
                        out[i][j] = self.data[i][j] / other[i][j]
                    elif isinstance(other, (int, float)):
                        out[i][j] = self.data[i][j] / other
        
        
        return out

    
    def __sub__(self, other):
        out = zeros(*self.shape)
        if len(self.shape) == 3:
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        out[b][i][j] = self.data[b][i][j] - other[b][i][j]
        else:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    out[i][j] = self.data[i][j] - other[i][j]
        
        return out

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    if len(self.shape) == 3:
                        for j in range(self.shape[2]):
                            self.data[b][i][j] = self.data[b][i][j] + other
                    else:
                            self.data[b][i] = self.data[b][i] + other

            
        else:
            assert isinstance(self, Tensor) and isinstance(other, Tensor) and self.shape == other.shape
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    if len(self.shape) == 3:
                        for j in range(self.shape[2]):
                            self.data[b][i][j] = self.data[b][i][j] + other[b][i][j]
                    else:
                            self.data[b][i] = self.data[b][i] + other[b][i]
       
        return self

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"Tensor({self.data})"

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __rpow__(self, other):
        out = zeros(*self.shape)
        if len(self.shape) == 3:
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        out[b][i][j] = other**(self.data[b][i][j])
        
        else:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    out[i][j] = other**(self.data[i][j])

        return Tensor(out.data)

    def transpose(self):
        if len(self.shape) == 3:
            out = zeros(self.shape[0], self.shape[-1], self.shape[1])
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        out[b][j][i] = self.data[b][i][j]

        elif len(self.shape) == 2:
            out = zeros(self.shape[1], self.shape[0])
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    out[j][i] = self.data[i][j]
        else:
            self.data
            out = Tensor(["Not working"])
        
        return Tensor(out.data)

    def item(self):
        assert sum([ax for ax in self.shape]) == len(self.shape) or len(self.shape) == 0
        if len(self.shape) == 0:
            return self.data

        if len(self.shape) == 1:
            return self.data[0]

        if len(self.shape) == 3:
            return self.data[0][0][0]
        else:
            return self.data[0][0]

    def get_shape(self, data):
        if not isinstance(data, list):
            return () # scalar

        return (len(data),) + self.get_shape(data[0])

    def unsqueeze(self):
        out = [self.data]
        return Tensor(out)
    
    def exp(self):
        # self**other
        e = 2.71828
        out = e**self

        return Tensor(out.data)

    def normalize(self):
        out = zeros(*self.shape)
        if len(self.shape) == 3:
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    row_sum = sum(self.data[b][i])
                    for j in range(self.shape[2]):
                        out[b][i][j] = self.data[b][i][j] / row_sum
        else:
            for i in range(self.shape[0]):
                row_sum = sum(self.data[i])
                for j in range(self.shape[1]):
                    out[i][j] = self.data[i][j] / row_sum
                        
        return Tensor(out.data)

    def flatten(self):
        if len(self.shape) == 3:
            out = zeros(self.shape[0], self.shape[1] * self.shape[2])
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        out[b][i * self.shape[2] + j] = self.data[b][i][j]
        else:
            out = []
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    out.append(self.data[i][j])
        
        return out if len(self.shape) == 3 else Tensor(out)

    def to_list(self):
        return self.data
    
    def atom_log(self):
        out = zeros(*self.shape)
        if len(self.shape) == 3:
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        out[b][i][j] = log(self.data[b][i][j])
        else:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    out[i][j] = log(self.data[i][j])

        return out

    def atom_sum(self):
        if len(self.shape) == 2:
            out = zeros(self.shape[0], 1)
            for i in range(self.shape[0]):
                out[i][0] = sum(self.data[i])

        else:
            out = zeros(self.shape[0], self.shape[1], 1)
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    out[b][i][0] = sum(self.data[b][i])

        return out
    
    def atom_mean(self):
        acc = 0
        if len(self.shape) == 2:
            for i in range(self.shape[0]):
                acc += self.data[i][0]
        return acc / self.shape[0]

    def backward(self):
        topo = []
        visited = set()

        def get_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    get_topo(child)
                
                topo.append(v)

        get_topo(self)

        for node in topo:
            node.grad = zeros(*node.shape)
        
        self.grad = ones(*self.shape)
        for node in reversed(topo):
            node._backward()


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.if_bias = bias
        self.weights = randn(in_features, out_features)
        self.bias = randn(1, 1, 1)[0][0][0]

    def __call__(self, other):
        return (other @ self.weights) + self.bias if self.if_bias else other @ self.weights
    
    def parameters(self):
        return [self.weights, self.bias] if self.if_bias else [self.weights]
            

class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        # parameters = [p1, p2, p3, ...]
        for param in self.parameters:
            param.grad = zeros(*param.shape)
    
    def step(self):
        for param in self.parameters:
            for b in range(param.shape[0]):
                for i in range(param.shape[1]):
                    for j in range(param.shape[2]):
                        param.data[b][i][j] -= self.lr * param.grad.data[b][i][j]


class CrossEntropyLoss:
    def __call__(self, logits, targets):
        assert targets.shape[-1] == logits.shape[-1]
        probs = softmax(logits)
        loss = -((targets * probs.atom_log()).atom_sum().atom_mean())
        
        def _backward():
            logits.grad = (probs - targets) / logits.shape[0]
        
        loss = Tensor([loss], (logits,))
        loss._backward = _backward

        return loss


class ReLU:
    def __call__(self, x):
        out = zeros(*x.shape)
        if len(x.shape) == 3:
            for b in range(x.shape[0]):
                for i in range(x.shape[1]):
                    for j in range(x.shape[2]):
                        if x[b][i][j] < 0:
                            out[b][i][j] = 0
                        else:
                            out[b][i][j] = x[b][i][j]
            
            def _backward():
                for b in range(x.shape[0]):
                    for i in range(x.shape[1]):
                        for j in range(x.shape[2]):
                            if x[b][i][j] > 0:
                                x.grad[b][i][j] += 1 * out.grad[b][i][j]
                

        else:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if x[i][j] < 0:
                        out[i][j] = 0
                    else:
                        out[i][j] = x[i][j]
        
            def _backward():
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        if x[i][j] > 0:
                            x.grad[i][j] += 1 * out.grad[i][j]
        
            
        out.children = {x}
        out._backward = _backward
        return out


def softmax(logits):
    logits = logits.exp()
    probs = logits.normalize()
    return probs

def random(seed, mod, a, c):
    while True:
        seed = (a * seed + c) % mod
        yield seed / mod

def randn(*shape):
    out = [[[None for _ in range(shape[-1])]for _ in range(shape[-2])] for _ in range(shape[0])] if len(shape) == 3 else [[None for _ in range(shape[-1])] for _ in range(shape[0])]
    gen = random(0, 2**32, 1664525, 1013904223)
    for b in range(shape[0]):
        for i in range(shape[1]):
            if len(shape) == 3:
                for j in range(shape[2]):
                    out[b][i][j] = next(gen)
            else:
                out[b][i] = next(gen)

    return Tensor(out)

def zeros(*shape):
    if len(shape) == 3:
        out = [[[0 for _ in range(shape[-1])]for _ in range(shape[-2])] for _ in range(shape[0])]
    else:
        out = [[0 for _ in range(shape[-1])] for _ in range(shape[0])]
    return Tensor(out)
    
def ones(*shape):
    if len(shape) == 3:
        out = [[[1 for _ in range(shape[-1])]for _ in range(shape[-2])] for _ in range(shape[0])]
    else:
        out = [[1 for _ in range(shape[-1])] for _ in range(shape[0])]
    return Tensor(out)


def one_hot(nums, num_classes):
    out = zeros(len(nums), num_classes)

    for i in range(len(nums)):
        out[i][nums[i]] = 1
    
    return out
