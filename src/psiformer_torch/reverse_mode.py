from typing import Any
import torch
from torch.autograd import Function


def cofactor_backward(A, C_bar, eps=1e-12):
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
    @staticmethod
    def forward(ctx, A):
        """
        A: (..., n, n)
        return C = Cof(A): (..., n, n)
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

        detU = torch.det(U)[..., None, None]
        detV = torch.det(V)[..., None, None]
        C = detU * detV * (U @ Gamma @ V.transpose(-2, -1))
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


def cofactor(A):
    return CofactorFn.apply(A)


class DetWithCofactor(Function):
    """
    Takes the Determinant from a matrix but
    the gradient of the determinant is our Cofactor
    Implementation from Google DeepMind.
    """
    @staticmethod
    def forward(ctx, A):
        det = torch.linalg.det(A)
        ctx.save_for_backward(A, det)
        return det

    @staticmethod
    def backward(ctx, grad_det):
        A, det = ctx.saved_tensors
        adj = cofactor(A)                # adjugate/cofactor
        grad_A = grad_det[..., None, None] * adj.transpose(-2, -1)
        return grad_A


class SLogDet(Function):
    """
    Makes exactly the same that the PyTorch Built-In "torch.linalg.slogdte"
    but it use our determinant implementation.
    """
    @staticmethod
    def forward(ctx, A, eps=1e-12):
        detA = DetWithCofactor.apply(A)
        sign = torch.sign(detA)
        logabs = torch.log(detA.abs().clamp_min(eps))
        ctx.save_for_backward(A, detA)
        ctx.eps = eps
        return sign, logabs

    @staticmethod
    def backward(ctx, grad_sign, grad_logabs):
        del grad_sign  # sign is not differentiable
        A, detA = ctx.saved_tensors
        if grad_logabs is None:
            return torch.zeros_like(A), None
        adj = cofactor(A)
        safe_abs = torch.where(
            detA.abs() < ctx.eps,
            detA.new_full(detA.shape, ctx.eps),
            detA.abs()
        )
        safe_sign = torch.where(
            detA == 0,
            torch.ones_like(detA),
            detA.sign()
        )
        safe_det = safe_sign * safe_abs
        grad_A = (
            grad_logabs[..., None, None]
            * adj.transpose(-2, -1)
            / safe_det[..., None, None]
        )
        return grad_A, None  # eps has no grad
