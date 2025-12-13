import torch

# Adjust this import to match your actual project structure
from psiformer_torch.backwards import SLogDet  # your custom autograd Function

torch.manual_seed(0)


def random_invertible(batch_shape, n, eps=1e-2):
    """
    Create a random (batch_shape, n, n) tensor that is *likely* invertible
    by adding eps * I.
    """
    A = torch.randn(*batch_shape, n, n)
    # Make it "more invertible-ish"
    I = torch.eye(n).expand(batch_shape + (n, n))
    return A + eps * I


def compare_for_shape(batch_shape, n=3, device="cpu", tol=1e-5):
    print(f"\n=== Testing shape {batch_shape + (n, n)} on {device} ===")

    A = random_invertible(batch_shape, n).to(device)
    A.requires_grad_(True)

    # ----- Reference: PyTorch slogdet -----
    A_ref = A.clone().detach().requires_grad_(True)
    sign_ref, logabs_ref = torch.linalg.slogdet(A_ref)
    loss_ref = logabs_ref.sum()
    loss_ref.backward()
    grad_ref = A_ref.grad

    # ----- Your custom SLogDet -----
    sign_custom, logabs_custom = SLogDet.apply(A)
    loss_custom = logabs_custom.sum()
    loss_custom.backward()
    grad_custom = A.grad

    # ----- Compare -----
    print("max |logabs_custom - logabs_ref|:",
          (logabs_custom - logabs_ref).abs().max().item())
    print("max |grad_custom - grad_ref|:",
          (grad_custom - grad_ref).abs().max().item())

    # Optional: assert-style checks
    ok_vals = torch.allclose(logabs_custom, logabs_ref, atol=tol, rtol=tol)
    ok_grads = torch.allclose(grad_custom, grad_ref, atol=tol, rtol=tol)
    print("values close?  ", ok_vals)
    print("grads close?   ", ok_grads)


if __name__ == "__main__":
    # Simple shapes
    compare_for_shape((), n=2)          # (2, 2)
    compare_for_shape((5,), n=3)        # (5, 3, 3)
    compare_for_shape((2, 4), n=3)      # (2, 4, 3, 3)  e.g. (B, n_det, n, n)

    # If you have CUDA:
    if torch.cuda.is_available():
        compare_for_shape((), n=2, device="cuda")
        compare_for_shape((5,), n=3, device="cuda")
        compare_for_shape((2, 4), n=3, device="cuda")
