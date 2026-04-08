import pytest
import torch
from src.atom import Tensor, Linear, randn

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