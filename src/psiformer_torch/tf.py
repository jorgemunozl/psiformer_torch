import torch


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
