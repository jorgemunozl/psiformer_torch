import torch

from psiformer_torch.logdet_matmul import logdet_matmul


def test_logdet_matmul_handles_near_singular_and_second_derivatives():
    """
    Determinant blocks can become close to singular during training.
    This ensures we still get finite first/second derivatives for the
    log-determinant aggregation used by the wavefunction.
    """
    # Two nearly linearly dependent rows -> near-singular determinant.
    base = torch.tensor([[1.0, 2.0], [2.0001, 4.0]], requires_grad=True)
    x1 = base.unsqueeze(0).unsqueeze(0)  # (B=1, n_det=1, 2, 2)
    x2 = torch.tensor([[[[1.0]]]], requires_grad=True)  # simple down-spin block
    w = torch.ones(1, 1)

    log_out, _ = logdet_matmul(x1, x2, w)
    assert torch.isfinite(log_out).all()

    grad_x1 = torch.autograd.grad(log_out.sum(), x1, create_graph=True)[0]
    assert torch.isfinite(grad_x1).all()

    second_x1 = torch.autograd.grad(grad_x1.sum(), x1, retain_graph=True)[0]
    assert torch.isfinite(second_x1).all()
