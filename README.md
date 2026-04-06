# Atomic Grad

Atomic Grad is a lightweight, educational deep learning framework focused on clarity, composability, and a small-but-powerful autograd core. It’s ideal for learning, research prototypes, and building minimal custom layers or optimizers.

Features
- Simple tensor abstraction with autodiff
- Eager execution and imperative API
- Modular layers and model composition
- Optimizers (SGD, Adam) and common losses
- Minimal dependencies; designed for readability and extensibility

Quickstart
1. Install (from repo):
```bash
git clone https://github.com/yourname/atomic_grad.git
cd atomic_grad
pip install -e .
```
2. Train a tiny model:
```py
from atomic_grad import Tensor, Linear, MSELoss, SGD

# toy data
x = Tensor([[0.0], [1.0], [2.0]])
y = Tensor([[0.0], [2.0], [4.0]])

model = Linear(1, 1)
opt = SGD(model.parameters(), lr=0.1)
loss_fn = MSELoss()

for epoch in range(200):
    preds = model(x)
    loss = loss_fn(preds, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

Core concepts
- Tensors: n-dimensional arrays that track operations to build a computation graph.
- Autograd: reverse-mode autodiff implemented with a small Function/Node API.
- Modules: composable layers exposing parameters() for optimization.
- Optimizers: simple implementations that update parameters using gradients.
- Losses/metrics: standard losses implemented in pure Python for clarity.

Extending
- Add new layers by subclassing Module and implementing forward.
- Add ops by composing existing tensor ops or creating new Function nodes for custom gradients.
- Implement custom optimizers by iterating model.parameters() and updating .data.

Contributing
- Open issues for bugs/features.
- Keep PRs small and well-documented; include tests for new ops or layers.

License
- MIT License — permissive and simple for research and experiments.

Roadmap (short)
- Additional ops and GPU support
- More optimizers and schedulers
- Serialization and CLI training utilities

Get in touch
- File issues or PRs on the repository for questions, ideas, or improvements.
- Keep examples minimal and focused on readability.