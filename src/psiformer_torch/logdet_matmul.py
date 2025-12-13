"""
Torch implementation of ferminet.networks.logdet_matmul.

This mirrors the TensorFlow custom-gradient version using a torch.autograd
Function so we keep the numerically-stable determinant handling based on SVD
while exposing both the log-abs value and its sign.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function

_EPS = 1e-20


def _extend(x: Tensor | float, y: Tensor) -> Tensor | float:
    """Add trailing dims to ``x`` so it can broadcast with ``y``."""
    if not torch.is_tensor(x):
        return x
    while x.dim() < y.dim():
        x = x.unsqueeze(-1)
    return x


def _gamma(s: Tensor, shift: Tensor | float = 0.0) -> Tensor:
    """Diagonal of Gamma with elements prod_{k!=i} s_k, computed safely."""
    log_s = torch.log(torch.clamp(s, min=_EPS))
    lower = torch.cumsum(log_s, dim=-1) - log_s
    upper = torch.cumsum(log_s.flip(-1), dim=-1) - log_s.flip(-1)
    upper = upper.flip(-1)
    return torch.exp(lower + upper - shift)


def _cofactor(u: Tensor, s: Tensor, v: Tensor,
              shift: Tensor | float = 0.0) -> Tensor:
    """Cofactor matrix up to det(U) * det(V)."""
    gamma = _gamma(s, shift)
    return u @ torch.diag_embed(gamma) @ v.transpose(-2, -1)


def _sign_det(x: Tensor) -> Tensor:
    """Sign of determinant with zero gradient almost everywhere."""
    return torch.sign(torch.linalg.det(x))


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

        # SVD-based determinants for numerical stability.
        u1, s1, v1h = torch.linalg.svd(x1, full_matrices=False)
        v1 = v1h.transpose(-2, -1)
        u2, s2, v2h = torch.linalg.svd(x2, full_matrices=False)
        v2 = v2h.transpose(-2, -1)

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
        log_out = torch.log(torch.abs(output) + _EPS) + logdet_max1 + logdet_max2

        ctx.save_for_backward(
            w, output, det, sign1, sign2, logdet1, logdet2, logdet_max1,
            logdet_max2, u1, s1, v1, u2, s2, v2
        )
        return log_out, sign_out

    @staticmethod
    def backward(ctx, grad_log: Tensor, grad_sign: Tensor):
        del grad_sign
        (
            w, output, det, sign1, sign2, logdet1, logdet2, logdet_max1,
            logdet_max2, u1, s1, v1, u2, s2, v2
        ) = ctx.saved_tensors

        glog_out = grad_log / output
        dout = glog_out @ w.t()

        adj1 = _cofactor(u1, s1, v1, _extend(logdet_max1, s1))
        adj2 = _cofactor(u2, s2, v2, _extend(logdet_max2, s2))
        adj1 = adj1 * sign1[..., None, None]
        adj2 = adj2 * sign2[..., None, None]

        det1 = torch.exp(logdet1 - logdet_max1) * sign1
        det2 = torch.exp(logdet2 - logdet_max2) * sign2

        dx1 = adj1 * (det2 * dout)[..., None, None]
        dx2 = adj2 * (det1 * dout)[..., None, None]

        dw = det.transpose(0, 1) @ glog_out

        return dx1, dx2, dw


def logdet_matmul(x1: Tensor, x2: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """Wrapper around LogDetMatmul for convenience."""
    return LogDetMatmul.apply(x1, x2, w)


__all__ = ["logdet_matmul", "LogDetMatmul"]
