from src.atom import Linear, ReLU, CrossEntropyLoss, one_hot, Tensor, SGD, softmax
from datasets import load_dataset
import numpy as np  
import random as rand
from tqdm import tqdm

ds = load_dataset("ylecun/mnist")

train_x, train_y = ds['train']['image'], ds['train']['label']

train_x = Tensor((np.array(train_x) / 255.0).tolist()).flatten()

test_x, test_y = ds['test']['image'], ds['test']['label']
test_x = Tensor((np.array(test_x) / 255.0).tolist()).flatten()

class MyNet:
    def __init__(self, in_dim, out_dim):
        self.l1 = Linear(in_dim, 128, bias=False)
        self.l2 = Linear(128, 128, bias=False)
        self.l3 = Linear(128, 64, bias=False)
        self.l4 = Linear(64, 64, bias=False)
        self.l5 = Linear(64, out_dim, bias=False)
        self.relu = ReLU()
    
    def __call__(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.l5(x)

        return x 
    
    def parameters(self):
        params = []
        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5]:
            params += layer.parameters()
        return params


def get_batch(test=False, bs=64):
    data = train_x if not test else test_x
    ix = rand.randint(0, data.shape[0] - (bs+1)) if not test else rand.randint(0, data.shape[0] - (bs+1))
    targets = one_hot(train_y[ix:ix+bs], 10) if not test else test_y[ix:ix+bs]
    
    return Tensor(data[ix:ix+bs]), targets

x, y = get_batch(test=True)

def evaluate():
    x, y = get_batch(test=True)
    correct = 0
    for i in range(x.shape[0]):
        inp = x[i]
        ans = y[i]
        logits = model(Tensor([inp]))
        probs = softmax(logits)
        out = probs.argmax()
        correct += 1 if out == ans else 0 
    
    return correct / x.shape[0]



model = MyNet(train_x.shape[1], 10)
optimizer = SGD(model.parameters(), lr=1e-1)
loss_fn = CrossEntropyLoss()
epochs = 1_000

x, y = get_batch()

for i in tqdm(range(epochs)):
    x, y = get_batch() 
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0 or (i + 1) == epochs:
        print(f"Loss at Epoch {i}: {loss.item()}")

