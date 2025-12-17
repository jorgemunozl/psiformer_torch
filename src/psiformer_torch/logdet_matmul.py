"""
Stable determinant aggregation with double-backward support.

We keep the SVD-based log|det| computation for numerical stability
but rely on PyTorch autograd for the derivatives so that second
derivatives (needed by the Laplacian) remain defined.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function

_EPS = 1e-12
_MIN_SINGULAR = 1e-6
_OUTPUT_FLOOR = 1e-12
_DET_JITTER = 1e-4


def _stabilize_matrix(x: Tensor, jitter: float) -> Tensor:
    """Add a small diagonal term to avoid exact singular matrices."""
    if jitter <= 0 or x.shape[-1] != x.shape[-2]:
        return x
    eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    eye = eye.view((1,) * (x.dim() - 2) + eye.shape)
    return x + jitter * eye


def _sign_det(x: Tensor) -> Tensor:
    """Sign of determinant with zero gradient almost everywhere."""
    return torch.sign(torch.linalg.det(x))


def _logdet_matmul_value(x1: Tensor, x2: Tensor,
                         w: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute log|sum_d w_d det(x1_d) det(x2_d)| in a numerically stable way.
    Returns (log_abs, sign).
    """
    x1 = _stabilize_matrix(x1, _DET_JITTER)
    x2 = _stabilize_matrix(x2, _DET_JITTER)

    # SVD-based determinants for stability; clamp small singular values.
    u1, s1_raw, v1h = torch.linalg.svd(x1, full_matrices=False)
    v1 = v1h.transpose(-2, -1)
    u2, s2_raw, v2h = torch.linalg.svd(x2, full_matrices=False)
    v2 = v2h.transpose(-2, -1)

    s1 = torch.clamp(s1_raw, min=_MIN_SINGULAR)
    s2 = torch.clamp(s2_raw, min=_MIN_SINGULAR)

    sign1 = _sign_det(u1) * _sign_det(v1)
    sign2 = _sign_det(u2) * _sign_det(v2)
    logdet1 = torch.sum(torch.log(torch.clamp(s1, min=_EPS)), dim=-1)
    logdet2 = torch.sum(torch.log(torch.clamp(s2, min=_EPS)), dim=-1)

    sign = sign1 * sign2
    logdet = logdet1 + logdet2

    logdet_max1, _ = logdet1.max(dim=-1, keepdim=True)
    logdet_max2, _ = logdet2.max(dim=-1, keepdim=True)

    det = torch.exp(logdet - logdet_max1 - logdet_max2) * sign
    output = det @ w

    sign_out = torch.sign(output)
    log_out = torch.log(torch.clamp(torch.abs(output), min=_OUTPUT_FLOOR))
    log_out += logdet_max1 + logdet_max2
    return log_out, sign_out


class LogDetMatmul(Function):
    """Stable log(abs(det(x1) * det(x2) @ w)) with sign tracking."""

    @staticmethod
    def forward(ctx, x1: Tensor, x2: Tensor, w: Tensor):
        if x1.shape[:-2] != x2.shape[:-2]:
            raise ValueError("x1 and x2 must share leading dimensions.")
        if w.shape[0] != x1.shape[-3]:
            raise ValueError(
                "Number of determinants must match w's first dimension."
            )

        log_out, sign_out = _logdet_matmul_value(x1, x2, w)
        ctx.save_for_backward(x1, x2, w)
        return log_out, sign_out

    @staticmethod
    def backward(ctx, grad_log: Tensor, grad_sign: Tensor):
        del grad_sign
        x1, x2, w = ctx.saved_tensors

        with torch.enable_grad():
            needs_x1, needs_x2, needs_w = ctx.needs_input_grad
            x1_ = x1 if needs_x1 else x1.detach()
            x2_ = x2 if needs_x2 else x2.detach()
            w_ = w if needs_w else w.detach()

            log_out, _ = _logdet_matmul_value(x1_, x2_, w_)

            inputs = []
            idx_map: list[str] = []
            if needs_x1:
                inputs.append(x1_)
                idx_map.append("x1")
            if needs_x2:
                inputs.append(x2_)
                idx_map.append("x2")
            if needs_w:
                inputs.append(w_)
                idx_map.append("w")

            grads = torch.autograd.grad(
                outputs=log_out,
                inputs=inputs,
                grad_outputs=grad_log,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            ) if inputs else ()

        # Reconstruct full gradient tuple in input order.
        dx1 = dx2 = dw = None
        for name, grad in zip(idx_map, grads):
            if name == "x1":
                dx1 = grad
            elif name == "x2":
                dx2 = grad
            elif name == "w":
                dw = grad

        return dx1, dx2, dw


def logdet_matmul(x1: Tensor, x2: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """Wrapper around LogDetMatmul for convenience."""
    return LogDetMatmul.apply(x1, x2, w)


__all__ = ["logdet_matmul", "LogDetMatmul"]
