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
        
        assert self.shape[0] == other.shape[0] and self.shape[-1] == other.shape[-2]

        # BxMxN @ BxNxK = BxMxK
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"Tensor({self.data})"

    def __getitem__(self, key):
        return self.data[key]
    
bs = 64

listt = [
    [[1 for _ in range(32)] for _ in range(32)] for _ in range(bs)
]

tensor = Tensor(listt)
print(tensor.shape)