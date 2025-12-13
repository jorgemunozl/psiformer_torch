from typing import Any
import torch
from torch.autograd import Function


def logdet_mul(x1, x2, w):  # From ferminet
    """Numerically stable implementation of
    log(abs(sum_i w_i |x1_i| * |x2_i|)).

    Args:
    x1: Tensor of shape [batch size, dets in, n, n].
    x2: Tensor of shape [batch size, dets in, m, m].
    w: Weight matrix of shape [dets in, dets out].

    Returns:
    Op that computes matmul(det(x1) * det(x2), w). For numerical stability,
    returns this op in the form of a tuple, the first of which is the log
    of the absolute value of the output, and the second of which is the sign of
    the output.

    Raises:
    ValueError if x1 and x2 do not have the same shape except the last two
    dimensions, or if the number of columns in det(x1)
    does not match the number of rows in w.
    """

    def logdet_matmul_grad(x1, x2, w, grad_log):
      """Numerically stable gradient of log(abs(sum_i w_i |x1_i| * |x2_i|)).

      Args:
        x1: Tensor of shape [batch size, dets in, n, n].
        x2: Tensor of shape [batch size, dets in, m, m].
        w: Weight matrix of shape [dets in, dets out].
        grad_log: Tensor of shape [batch_size, dets_out] by which to
          left-multiply the Jacobian (aka the reverse sensitivity).

      Returns:
        Ops that compute the gradient of log(abs(matmul(det(x1) * det(x2), w))).
        The ops give the gradients with respect to the inputs x1, x2, w.
      """
      # This is missing a factor of exp(-logdet_max1-logdet_max2), which is
      # instead picked up by including it in the terms adj1*det2 and adj2*det1
      glog_out = grad_log / output
      dout = tf.matmul(glog_out, w, transpose_b=True)

      if x1.shape[-1] > 1:
        adj1 = cofactor(u1, s1, v1, extend(logdet_max1, s1)) * sign1[..., None,
                                                                     None]
      else:
        adj1 = tf.ones_like(x1) * extend(tf.exp(-logdet_max1), x1)

      if x2.shape[-1] > 1:
        adj2 = cofactor(u2, s2, v2, extend(logdet_max2, s2)) * sign2[..., None,
                                                                     None]
      else:
        adj2 = tf.ones_like(x2) * extend(tf.exp(-logdet_max2), x2)

      det1 = tf.exp(logdet1 - logdet_max1) * sign1
      det2 = tf.exp(logdet2 - logdet_max2) * sign2

      dx1 = adj1 * (det2 * dout)[..., None, None]
      dx2 = adj2 * (det1 * dout)[..., None, None]
      dw = tf.matmul(det, glog_out, transpose_a=True)

      def grad(grad_dx1, grad_dx2, grad_dw):
        """Stable gradient of gradient of log(abs(sum_i w_i |x1_i| * |x2_i|)).

        Args:
          grad_dx1: Tensor of shape [batch size, dets in, n, n].
          grad_dx2: Tensor of shape [batch size, dets in, m, m].
          grad_dw: Tensor of shape [dets in, dets out].

        Returns:
          Ops that compute the gradient of the gradient of
          log(abs(matmul(det(x1) * det(x2), w))). The ops give the gradients
          with respect to the outputs of the gradient op dx1, dx2, dw and the
          reverse sensitivity grad_log.
        """
        # Terms that appear repeatedly in different gradients.
        det_grad_dw = tf.matmul(det, grad_dw)
        # Missing a factor of exp(-2logdet_max1-2logdet_max2) which is included\
        # via terms involving r1, r2, det1, det2, adj1, adj2
        glog_out2 = grad_log / (output**2)

        adj_dx1 = tf.reduce_sum(adj1 * grad_dx1, axis=[-1, -2])
        adj_dx2 = tf.reduce_sum(adj2 * grad_dx2, axis=[-1, -2])

        adj_dx1_det2 = det2 * adj_dx1
        adj_dx2_det1 = det1 * adj_dx2
        adj_dx = adj_dx2_det1 + adj_dx1_det2

        if x1.shape[-1] > 1:
          r1 = rho(s1, logdet_max1)
          dadj1 = (grad_cofactor(u1, v1, r1, grad_dx1) * sign1[..., None, None])
        else:
          dadj1 = tf.zeros_like(x1)

        if x2.shape[-1] > 1:
          r2 = rho(s2, logdet_max2)
          dadj2 = (grad_cofactor(u2, v2, r2, grad_dx2) * sign2[..., None, None])
        else:
          dadj2 = tf.zeros_like(x2)

        # Computes gradients wrt x1 and x2.
        ddout_w = (
            tf.matmul(glog_out, grad_dw, transpose_b=True) -
            tf.matmul(glog_out2 * det_grad_dw, w, transpose_b=True))
        ddout_x = tf.matmul(
            glog_out2 * tf.matmul(adj_dx, w), w, transpose_b=True)

        grad_x1 = (
            adj1 * (det2 *
                    (ddout_w - ddout_x) + adj_dx2 * dout)[..., None, None] +
            dadj1 * (dout * det2)[..., None, None])
        grad_x2 = (
            adj2 * (det1 *
                    (ddout_w - ddout_x) + adj_dx1 * dout)[..., None, None] +
            dadj2 * (dout * det1)[..., None, None])

        adj_dx_w = tf.matmul(adj_dx, w)

        # Computes gradient wrt w.
        grad_w = (
            tf.matmul(adj_dx, glog_out, transpose_a=True) - tf.matmul(
                det, glog_out2 * (det_grad_dw + adj_dx_w), transpose_a=True))

        # Computes gradient wrt grad_log.
        grad_grad_log = (det_grad_dw + adj_dx_w) / output
        return grad_x1, grad_x2, grad_w, grad_grad_log

      return (dx1, dx2, dw), grad


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
    def forward(ctx, A: torch.Tensor) -> torch.Tensor:
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

        detU = torch.linalg.det(U)
        detV = torch.linalg.det(V)

        prefac = (detU * detV)[..., None, None]

        C = prefac * (U @ Gamma @ V.transpose(-2, -1))
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
    def forward(ctx, A: torch.Tensor):
        # detA: shape (...,)
        detA = DetWithCofactor.apply(A)
        sign, logabs = detA.sign(), torch.log(detA.abs()+1e-20)
        ctx.save_for_backward(A, detA)
        return sign, logabs

    @staticmethod
    def backward(ctx, grad_sign, grad_logabs):
        A, detA = ctx.saved_tensors
        print(A)
        del grad_sign

        C = CofactorFn.apply(A)

        print("GRAD_LOGABS", grad_logabs)
        print("DET", detA)


        print("COFACTOR", C)
        factor = grad_logabs[..., None, None] / detA[..., None, None]
        grad_A = factor * C

        return grad_A
