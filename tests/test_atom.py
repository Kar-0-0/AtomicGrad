import pytest
import torch
import math
from src.atom import Tensor, Linear, randn, softmax, CrossEntropyLoss, ReLU

def test_matmul_basic():
    # create the same data in both frameworks
    data1 = torch.randn(2, 3, 4)
    data2 = torch.randn(2, 4, 5)
    
    torch_result = (data1 @ data2).flatten().tolist()
    
    my1 = Tensor(data1.tolist())
    my2 = Tensor(data2.tolist())
    my_result = torch.tensor((my1 @ my2).data).flatten()
    
    assert torch.allclose(torch.tensor(torch_result), my_result, atol=1e-5)

def test_matmul_shape():
    my1 = Tensor(torch.randn(2, 3, 4).tolist())
    my2 = Tensor(torch.randn(2, 4, 5).tolist())
    result = my1 @ my2
    assert result.shape == (2, 3, 5)

def test_matmul_shape_mismatch():
    my1 = Tensor(torch.randn(2, 3, 4).tolist())
    my2 = Tensor(torch.randn(2, 3, 5).tolist())
    with pytest.raises(AssertionError):
        my1 @ my2

def test_biger_mat_mul():
    data1 = torch.randn(64, 32, 32)
    data2 = torch.randn(64, 32, 32)
    
    torch_result = (data1 @ data2).flatten().tolist()
    
    my1 = Tensor(data1.tolist())
    my2 = Tensor(data2.tolist())
    my_result = torch.tensor((my1 @ my2).data).flatten()
    
    assert torch.allclose(torch.tensor(torch_result), my_result, atol=1e-5)

def test_linear_output_shape_no_bias():
    layer = Linear(4, 8, bias=False)
    x = randn(2, 3, 4)
    out = layer(x)
    assert out.shape == (2, 3, 8)

def test_linear_output_shape_with_bias():
    layer = Linear(4, 8, bias=True)
    x = randn(2, 3, 4)
    out = layer(x)
    assert out.shape == (2, 3, 8)

def test_linear_output_is_tensor():
    layer = Linear(4, 8, bias=False)
    x = randn(2, 3, 4)
    out = layer(x)
    assert isinstance(out, Tensor)

def test_linear_different_in_out_features():
    layer = Linear(16, 32, bias=False)
    x = randn(4, 10, 16)
    out = layer(x)
    assert out.shape == (4, 10, 32)

def test_linear_wrong_input_features_raises():
    layer = Linear(4, 8, bias=False)
    x = randn(2, 3, 6)  # 6 != 4
    with pytest.raises(AssertionError):
        layer(x)

def test_softmax():
    x = Tensor([[1.0, 2.0]])
    result = softmax(x)
    
    e1 = math.exp(1.0)
    e2 = math.exp(2.0)
    expected = [e1 / (e1 + e2), e2 / (e1 + e2)]
    
    assert abs(result[0][0] - expected[0]) < 1e-4
    assert abs(result[0][1] - expected[1]) < 1e-4    

def test_softmax_batched():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    result = softmax(x)
    
    e1, e2 = math.exp(1.0), math.exp(2.0)
    e3, e4 = math.exp(3.0), math.exp(4.0)
    
    expected = [
        [e1 / (e1 + e2), e2 / (e1 + e2)],
        [e3 / (e3 + e4), e4 / (e3 + e4)]
    ]
    
    for i in range(2):
        for j in range(2):
            assert abs(result[i][j] - expected[i][j]) < 1e-4

def test_softmax_large_batch():
    import random
    random.seed(42)
    
    data = [[random.uniform(-5, 5) for _ in range(10)] for _ in range(64)]
    
    x = Tensor(data)
    result = softmax(x)
    
    torch_result = torch.softmax(torch.tensor(data), dim=-1)
    
    for i in range(64):
        for j in range(10):
            assert abs(result[i][j] - torch_result[i][j].item()) < 1e-4

def test_softmax_3d():
    import random
    random.seed(42)
    
    data = [[[random.uniform(-5, 5) for _ in range(10)] for _ in range(4)] for _ in range(64)]
    
    x = Tensor(data)
    result = softmax(x)
    
    torch_result = torch.softmax(torch.tensor(data), dim=-1)
    
    for b in range(64):
        for i in range(4):
            for j in range(10):
                assert abs(result[b][i][j] - torch_result[b][i][j].item()) < 1e-4

def test_matmul_grad():
    import random
    random.seed(42)
    
    data_a = [[[random.uniform(-5, 5) for _ in range(4)] for _ in range(3)] for _ in range(2)]
    data_b = [[[random.uniform(-5, 5) for _ in range(5)] for _ in range(4)] for _ in range(2)]
    
    a = Tensor(data_a)
    b = Tensor(data_b)
    
    out = a @ b
    out.backward()
    
    ta = torch.tensor(data_a, requires_grad=True)
    tb = torch.tensor(data_b, requires_grad=True)
    
    tout = ta @ tb
    tout.sum().backward()
    
    for bi in range(2):
        for i in range(3):
            for j in range(4):
                assert abs(a.grad.data[bi][i][j] - ta.grad[bi][i][j].item()) < 1e-4
    
    for bi in range(2):
        for i in range(4):
            for j in range(5):
                assert abs(b.grad.data[bi][i][j] - tb.grad[bi][i][j].item()) < 1e-4

def test_add_grad():
    import random
    random.seed(42)
    
    data_a = [[[random.uniform(-5, 5) for _ in range(4)] for _ in range(3)] for _ in range(2)]
    data_b = [[[random.uniform(-5, 5) for _ in range(4)] for _ in range(3)] for _ in range(2)]
    
    a = Tensor(data_a)
    b = Tensor(data_b)
    
    out = a + b
    out.backward()
    
    ta = torch.tensor(data_a, requires_grad=True)
    tb = torch.tensor(data_b, requires_grad=True)
    
    tout = ta + tb
    tout.sum().backward()
    
    for bi in range(2):
        for i in range(3):
            for j in range(4):
                assert abs(a.grad.data[bi][i][j] - ta.grad[bi][i][j].item()) < 1e-4
                assert abs(b.grad.data[bi][i][j] - tb.grad[bi][i][j].item()) < 1e-4

def test_matmul_grad_large():
    import random
    random.seed(42)
    
    data_a = [[[random.uniform(-5, 5) for _ in range(32)] for _ in range(32)] for _ in range(64)]
    data_b = [[[random.uniform(-5, 5) for _ in range(32)] for _ in range(32)] for _ in range(64)]
    
    a = Tensor(data_a)
    b = Tensor(data_b)
    
    out = a @ b
    out.backward()
    
    ta = torch.tensor(data_a, requires_grad=True)
    tb = torch.tensor(data_b, requires_grad=True)
    
    tout = ta @ tb
    tout.sum().backward()
    
    for bi in range(64):
        for i in range(32):
            for j in range(32):
                assert abs(a.grad.data[bi][i][j] - ta.grad[bi][i][j].item()) < 1e-4
                assert abs(b.grad.data[bi][i][j] - tb.grad[bi][i][j].item()) < 1e-4


def test_add_grad_large():
    import random
    random.seed(42)
    
    data_a = [[[random.uniform(-5, 5) for _ in range(32)] for _ in range(32)] for _ in range(64)]
    data_b = [[[random.uniform(-5, 5) for _ in range(32)] for _ in range(32)] for _ in range(64)]
    
    a = Tensor(data_a)
    b = Tensor(data_b)
    
    out = a + b
    out.backward()
    
    ta = torch.tensor(data_a, requires_grad=True)
    tb = torch.tensor(data_b, requires_grad=True)
    
    tout = ta + tb
    tout.sum().backward()
    
    for bi in range(64):
        for i in range(32):
            for j in range(32):
                assert abs(a.grad.data[bi][i][j] - ta.grad[bi][i][j].item()) < 1e-4
                assert abs(b.grad.data[bi][i][j] - tb.grad[bi][i][j].item()) < 1e-4

def test_mul():
    import random
    random.seed(42)
    
    data_a = [[random.uniform(-5, 5) for _ in range(10)] for _ in range(64)]
    data_b = [[random.uniform(-5, 5) for _ in range(10)] for _ in range(64)]
    
    a = Tensor(data_a)
    b = Tensor(data_b)
    
    result = a * b
    
    ta = torch.tensor(data_a)
    tb = torch.tensor(data_b)
    torch_result = ta * tb
    
    for i in range(64):
        for j in range(10):
            assert abs(result[i][j] - torch_result[i][j].item()) < 1e-4


def test_log():
    import random
    random.seed(42)
    
    data = [[random.uniform(0.1, 5) for _ in range(10)] for _ in range(64)]
    
    a = Tensor(data)
    result = a.atom_log()
    
    ta = torch.tensor(data)
    torch_result = torch.log(ta)
    
    for i in range(64):
        for j in range(10):
            assert abs(result[i][j] - torch_result[i][j].item()) < 1e-4


def test_sum():
    import random
    random.seed(42)
    
    data = [[random.uniform(-5, 5) for _ in range(10)] for _ in range(64)]
    
    a = Tensor(data)
    result = a.atom_sum()
    
    ta = torch.tensor(data)
    torch_result = torch.sum(ta, dim=-1)
    
    for i in range(64):
        assert abs(result[i][0] - torch_result[i].item()) < 1e-4


def test_mean():
    import random
    random.seed(42)
    
    data = [[random.uniform(-5, 5)] for _ in range(64)]
    
    a = Tensor(data)
    result = a.atom_mean()
    
    ta = torch.tensor(data)
    torch_result = torch.mean(ta)
    
    assert abs(result - torch_result.item()) < 1e-4

def test_ce_loss():
    import random
    random.seed(42)
    
    data_logits = [[random.uniform(-5, 5) for _ in range(10)] for _ in range(64)]
    data_targets = [[0 for _ in range(10)] for _ in range(64)]
    
    for i in range(64):
        target_class = random.randint(0, 9)
        data_targets[i][target_class] = 1
    
    logits = Tensor(data_logits)
    targets = Tensor(data_targets)
    
    loss = CrossEntropyLoss()(logits, targets)

    
    torch_logits = torch.tensor(data_logits, dtype=torch.float32)
    torch_targets = torch.tensor([data_targets[i].index(1) for i in range(64)])
    
    torch_loss = torch.nn.CrossEntropyLoss()(torch_logits, torch_targets)
    
    print(loss.shape)
    assert abs(loss.item() - torch_loss.item()) < 1e-4

def test_ce_loss_backward():
    import random
    random.seed(42)
    
    data_logits = [[random.uniform(-5, 5) for _ in range(10)] for _ in range(64)]
    data_targets = [[0 for _ in range(10)] for _ in range(64)]
    
    for i in range(64):
        target_class = random.randint(0, 9)
        data_targets[i][target_class] = 1
    
    logits = Tensor(data_logits)
    targets = Tensor(data_targets)
    
    loss = CrossEntropyLoss()(logits, targets)
    loss.backward()
    
    torch_logits = torch.tensor(data_logits, dtype=torch.float32, requires_grad=True)
    torch_targets = torch.tensor([data_targets[i].index(1) for i in range(64)])
    
    torch_loss = torch.nn.CrossEntropyLoss()(torch_logits, torch_targets)
    torch_loss.backward()
    
    for i in range(64):
        for j in range(10):
            assert abs(logits.grad[i][j] - torch_logits.grad[i][j].item()) < 1e-4

def test_relu_grad():
    import random
    random.seed(42)
    
    data = [[random.uniform(-5, 5) for _ in range(10)] for _ in range(64)]
    
    x = Tensor(data)
    out = ReLU()(x)
    out.backward()
    
    tx = torch.tensor(data, requires_grad=True)
    tout = torch.relu(tx)
    tout.sum().backward()
    
    for i in range(64):
        for j in range(10):
            assert abs(x.grad[i][j] - tx.grad[i][j].item()) < 1e-4