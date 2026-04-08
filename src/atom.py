class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.shape = self.get_shape(data)
        self.children = set(_children)
        self.op = _op

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

        return Tensor(out.data, (self, other), '@')

    def __add__(self, other):
        out = zeros(*self.shape)
        if isinstance(other, (int, float)):
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                        out[b][i][j] = self.data[b][i][j] + other
            
            return Tensor(out.data, (self, other), "+")

        else:
            assert isinstance(self, Tensor) and isinstance(other, Tensor) and self.shape == other.shape
            for b in range(self.shape[0]):
                for i in range(self.shape[1]):
                    for j in range(self.shape[2]):
                         out[b][i][j] = self.data[b][i][j] + other[b][i][j]

            return Tensor(out, (self, other), "+")

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
    
    def broadcast(self, bs):
        out = [self.data[0] for _ in range(bs)]
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
    out = [[[0 for _ in range(shape[-1])]for _ in range(shape[-2])] for _ in range(shape[0])]

    return Tensor(out)
    
def ones(*shape):
    out = [[[1 for _ in range(shape[-1])]for _ in range(shape[-2])] for _ in range(shape[0])]

    return Tensor(out)

