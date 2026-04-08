class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.shape = self.get_shape(data)
        self.children = set(_children)
        self.op = _op
        self.grad = None
        self._backward = lambda: None

    def get_shape(self, data):
        if not isinstance(data, list):
            return () # scalar

        return (len(data),) + self.get_shape(data[0])

    def __matmul__(self, other):
        if not isinstance(self, Tensor) or not isinstance(other, Tensor):
            raise TypeError("Both values must be Tensors")

        # BxMxN @ BxNxK = BxMxK
        assert self.shape[-1] == other.shape[-2]
        if len(other.shape) == 3:
            assert self.shape[0] == other.shape[0] 
            
        B, M, K = self.shape[0], self.shape[1], other.shape[-1]
        out = zeros(B, M, K)
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

        def _backward():
            self.grad += out.grad.__matmul__(other.transpose()) 
            other.grad += self.transpose().__matmul__(out.grad)  

        out._backward = _backward

        return Tensor(out.data, (self, other), '@')

    def __add__(self, other):
        out = zeros(*self.shape)
        if isinstance(other, (int, float)):
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        out[b][i][j] = self.data[b][i][j] + other
            
        else:
            assert isinstance(self, Tensor) and isinstance(other, Tensor) and self.shape == other.shape
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                         out[b][i][j] = self.data[b][i][j] + other[b][i][j]
        
        def _backward():
            self.grad += out.grad 
            if isinstance(other, (Tensor)):
                other.grad += out.grad

        out._backward = _backward
    
        return Tensor(out.data, (self, other), "+")

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
            print(self.shape)
            self.data
            out = Tensor(["Not working"])
        
        return Tensor(out.data)
    
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

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        self.data[b][i][j] = self.data[b][i][j] + other
            
        else:
            assert isinstance(self, Tensor) and isinstance(other, Tensor) and self.shape == other.shape
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                         self.data[b][i][j] = self.data[b][i][j] + other.data[b][i][j]
        

        return self

        

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"Tensor({self.data})"

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def unsqueeze(self):
        out = [self.data]
        return Tensor(out)


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.if_bias = bias
        self.weights = randn(in_features, out_features)
        self.bias = randn(1, 1, 1)[0][0][0]

    def __call__(self, other):
        return (other @ self.weights) + self.bias if self.if_bias else other @ self.weights


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

