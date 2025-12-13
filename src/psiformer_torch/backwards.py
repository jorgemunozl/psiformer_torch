from typing import Any
import torch
from torch.autograd import Function


def cofactor_backward(A, C_bar, eps=1e-12):
    """
    C_bar comes from chain rule
    Implementation of the backward sensivity from A.
    Return A_bar, which is useful for backpropagation.
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    V = Vh.transpose(-2, -1)
    n = S.shape[-1]

    # Build Gamma safely
    Gamma = torch.zeros_like(A)
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool, device=A.device)
        mask[i] = False
        Gamma[..., i, i] = torch.prod(S[..., mask], dim=-1)

    # Safe inverse of Sigma
    S_safe = torch.where(S.abs() < eps, S.new_full((), eps), S)
    Sigma_inv = torch.diag_embed(1.0 / S_safe)

    # Reverse-mode formula
    M = V.transpose(-2, -1) @ C_bar.transpose(-2, -1) @ U
    MG = M @ Gamma
    Tr_MG = torch.einsum("...ii->...", MG)
    Xi = Tr_MG[..., None, None] * Sigma_inv - Sigma_inv @ MG
    A_bar = U @ Xi @ V.transpose(-2, -1)
    return A_bar


class CofactorFn(Function):
    """
    If returns the Cofactor and A_bar, where the determinant
    appears.
    """
    @staticmethod
    def forward(ctx, A):
        """
        A: (..., n, n)
        return Cofactor Matrix of A safely. Torch doesn't have
        a Cof(A) built-in.
        C = Cof(A): (..., n, n)
        """
        # SVD
        U, S, Vh = torch.linalg.svd(A, full_matrices=True)
        V = Vh.transpose(-2, -1)

        # Gamma diag:
        n = S.shape[-1]
        Gamma = torch.zeros_like(A)

        for i in range(n):
            mask = torch.ones(n, dtype=torch.bool, device=A.device)
            mask[i] = False
            Gamma[..., i, i] = torch.prod(S[..., mask], dim=-1)

        C = torch.det(U) * torch.det(V) * (U @ Gamma @ V.transpose(-2, -1))
        ctx.save_for_backward(A)
        return C

    @staticmethod
    def backward(ctx: Any, C_bar: Any) -> Any:
        """
        C_bar: dL/dC same shape as C
        return:(dL/dA, )
        """
        (A, ) = ctx.saved_tensors
        A_bar = cofactor_backward(A, C_bar)
        return A_bar


class DetWithCofactor(Function):
    """
    Takes the Determinant from a matrix but
    the gradient of the determinant is our Cofactor

    A_Bar?
    Implementation from Google DeepMind.
    """
    @staticmethod
    def forward(ctx, A):
        det = torch.linalg.det(A)
        ctx.save_for_backward(A)
        return det

    @staticmethod
    def backward(ctx, grad_det):
        (A,) = ctx.saved_tensors
        grad_A = grad_det[..., None, None] * CofactorFn.apply(A)
        return grad_A


class SLogDet(Function):
    """
    Makes exactly the same that the PyTorch Built-In "torch.linalg.slogdte"
    but it use our determinant implementation.
    """
    @staticmethod
    def forward(ctx, A, eps=1e-12):

        detA = DetWithCofactor.apply(A)
        sign, logDetA = A.sign(), torch.log(detA.abs())
        # to ignore the slogdet built i
        ctx.save_for_backward(A, sign)
        ctx.eps = eps
        return sign, logDetA

    @staticmethod
    def backward(ctx, grad_logabs, grad_sign):
        """
        This is not automatic? I don't know!!
        """
        A, sign = ctx.saved_tensors
        del grad_sign  # not differentiable
        det = torch.linalg.det(A)
        safe_det = torch.where(det.abs() < ctx.eps, det.sign() * ctx.eps, det)
        grad_A = grad_logabs[..., None, None] * adj.transpose(-2, -1) / safe_det[..., None, None]
        return grad_A, None  # eps has no grad


def logdet_matmul(x1: torch.Tensor, x2: torch.Tensor,
                  w: torch.Tensor, eps: float = 1e-12):
    """
    Compute log(|det(x1) * det(x2)| @ w) in a numerically stable way.

    Args:
        x1, x2: tensors with shape [B, D, n, n]
        w: weight matrix with shape [D, Dout]
        eps: clamp to avoid log(0)
    Returns:
        log_out, sign_out with shape [B, Dout]
    """
    sign1, logdet1 = torch.linalg.slogdet(x1)         # [B, D]
    sign2, logdet2 = torch.linalg.slogdet(x2)         # [B, D]

    logdet_max1 = logdet1.max(dim=-1, keepdim=True).values
    logdet_max2 = logdet2.max(dim=-1, keepdim=True).values

    det1 = torch.exp(logdet1 - logdet_max1) * sign1   # [B, D]
    det2 = torch.exp(logdet2 - logdet_max2) * sign2   # [B, D]

    det = det1 * det2                                 # [B, D]
    output = det @ w                                  # [B, Dout]

    log_out = torch.log(torch.clamp(output.abs(), min=eps)) + logdet_max1 + logdet_max2
    sign_out = output.sign()
    return log_out, sign_out
