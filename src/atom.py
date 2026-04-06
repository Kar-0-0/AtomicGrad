from random import random

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
        assert self.shape[0] == other.shape[0] and self.shape[-1] == other.shape[-2]
        B, M, K = self.shape[0], self.shape[1], other.shape[-1]

        out = [[[None for _ in range(K)]for _ in range(M)] for _ in range(B)] # Empty list of shape (B, M, K)
        


    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"Tensor({self.data})"

    def __getitem__(self, key):
        return self.data[key]
    