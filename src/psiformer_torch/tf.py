import torch
from torch.autograd import Function

def cofactor_from_svd(u, s, v, shift=0.0):
    # Gamma_ii = prod_{k!=i} s_k, done in log-domain then divided by exp(shift)
    ls = s.log()
    lower = torch.cumsum(ls, dim=-1) - ls           # exclusive prefix
    upper = torch.cumsum(ls.flip(-1), dim=-1).flip(-1) - ls  # exclusive suffix
    gamma = torch.exp(lower + upper - shift.unsqueeze(-1))
    gamma_mat = torch.diag_embed(gamma)
    return u @ gamma_mat @ v.transpose(-2, -1)

class LogdetMatmulFn(Function):
    @staticmethod
    def forward(ctx, x1, x2, w, eps=1e-12):
        # x1, x2: [B, D, n, n], w: [D, Dout]
        # torch.linalg.svd returns U, S, Vh; keep that ordering so the
        # singular values stay in s* variables.
        u1, s1, v1h = torch.linalg.svd(x1, full_matrices=False)
        u2, s2, v2h = torch.linalg.svd(x2, full_matrices=False)
        v1 = v1h.transpose(-2, -1)
        v2 = v2h.transpose(-2, -1)

        # Handle potentially rectangular orbital blocks by taking the square
        # k x k (k = min(m, n)) factor from the orthonormal bases.
        k1 = s1.shape[-1]
        k2 = s2.shape[-1]
        sign1 = torch.det(u1[..., :k1, :k1]) * torch.det(v1[..., :k1, :k1])
        sign2 = torch.det(u2[..., :k2, :k2]) * torch.det(v2[..., :k2, :k2])

        logdet1 = s1.log().sum(dim=-1)          # shape [B, D]
        logdet2 = s2.log().sum(dim=-1)

        logdet_max1 = logdet1.max(dim=-1, keepdim=True).values  # [B,1]
        logdet_max2 = logdet2.max(dim=-1, keepdim=True).values

        det1 = torch.exp(logdet1 - logdet_max1) * sign1         # [B,D]
        det2 = torch.exp(logdet2 - logdet_max2) * sign2
        det = det1 * det2                                       # [B,D]

        output = det @ w                                        # [B, Dout]
        log_out = torch.log(torch.clamp(output.abs(), min=eps)) + logdet_max1 + logdet_max2
        sign_out = output.sign()

        ctx.save_for_backward(u1, s1, v1, u2, s2, v2, sign1, sign2,
                              logdet_max1, logdet_max2, det, w, output)
        ctx.eps = eps
        return log_out, sign_out

    @staticmethod
    def backward(ctx, grad_log, grad_sign):
        u1, s1, v1, u2, s2, v2, sign1, sign2, logdet_max1, logdet_max2, det, w, output = ctx.saved_tensors
        del grad_sign
        eps = ctx.eps

        glog_out = grad_log / torch.clamp(output, min=eps)       # [B, Dout]
        dout = glog_out @ w.T                                    # [B, D]

        adj1 = cofactor_from_svd(u1, s1, v1, shift=logdet_max1) * sign1[..., None, None]
        adj2 = cofactor_from_svd(u2, s2, v2, shift=logdet_max2) * sign2[..., None, None]

        det1 = torch.exp(s1.log().sum(-1) - logdet_max1) * sign1
        det2 = torch.exp(s2.log().sum(-1) - logdet_max2) * sign2

        dx1 = adj1 * (det2 * dout)[..., None, None]
        dx2 = adj2 * (det1 * dout)[..., None, None]
        dw = det.transpose(-2, -1) @ glog_out                    # [D, Dout]
        return dx1, dx2, dw, None

logdet_matmul = LogdetMatmulFn.apply
