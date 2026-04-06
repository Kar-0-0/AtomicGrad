from random import random

class Tensor:
    def __init__(self, data, _children=(), _op='', *shape):
        self.data = data
        self.shape = shape if not isinstance(shape[0], tuple) else shape[0]
        self.children = set(_children)
        self.op = _op

    def __matmul__(self, other):
        if not isinstance(self, Tensor) or not isinstance(other, Tensor):
            raise TypeError("Both values must be Tensors")
        
        B, M, N = self.shape
        B2, N2, K = other.shape

        assert B == B2 and N == N2

        # BxMxN @ BxNxK = BxMxK
        out = []
        for b in range(B):
            mat1_ix = b * M * N
            mat2_ix = b * N * K 
            for i in range(M):
                for j in range(K):
                    acc = 0
                    for k in range(N):
                        acc += self.data[mat1_ix + (i * N + k)] * other.data[mat2_ix + (k * K + j)]

                    out.append(acc) 
         
        return Tensor(out, (self, other), "@", B, M, K)
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"Tensor({self.data})"

    def __getitem__(self, key):
        return self.data[key]

def rand(*shape):
    prod = 1
    for ax in shape:
        prod *= ax
    
    out = []
    for _ in range(prod):
        num = random()
        out.append(num)
    
    return Tensor(out, (), None, shape)

