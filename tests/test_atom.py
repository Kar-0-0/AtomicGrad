import pytest
import torch
import math
from src.atom import Tensor, Linear, randn, softmax

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