"""Compare jacfwd vs jacrev on the full Jacobian dy/dx for small MLPs."""
import time

import torch
import torch.nn as nn
from torch.func import functional_call, jacfwd, jacrev


HIDDEN = 128
CASES = [
    ("few_inputs_many_outputs", 10, 1000),
    ("many_inputs_few_outputs", 1000, 10),
]


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = HIDDEN, d_out: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


def make_pure_forward(model: nn.Module):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def forward_only(x_: torch.Tensor) -> torch.Tensor:
        return functional_call(model, (params, buffers), (x_,))

    return forward_only


def full_jacobian_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx with forward mode."""
    return jacfwd(make_pure_forward(model))(x)


def full_jacobian_reverse(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx with reverse mode."""
    return jacrev(make_pure_forward(model))(x)


def scalar_grad_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute d(sum(y))/dx with forward mode via full Jacobian row summation."""
    return full_jacobian_forward(model, x).sum(dim=-2)


def scalar_grad_reverse(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute d(sum(y))/dx with reverse mode."""
    x = x.detach().clone().requires_grad_(True)
    model(x).sum().backward()
    return x.grad


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(name: str, fn, n_warmup: int = 10, n_iter: int = 50) -> float:
    for _ in range(n_warmup):
        fn()
    _sync_if_cuda()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    _sync_if_cuda()
    return (time.perf_counter() - t0) / n_iter


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Full Jacobian dy/dx benchmark")
    print("Forward mode cost scales with number of inputs.")
    print("Reverse mode cost scales with number of outputs.")
    print("")

    for name, d_in, d_out in CASES:
        model = MLP(d_in=d_in, d_out=d_out).to(device)
        x = torch.randn(d_in, device=device)

        jac_fwd = full_jacobian_forward(model, x)
        jac_rev = full_jacobian_reverse(model, x)
        assert torch.allclose(jac_fwd, jac_rev, rtol=1e-4, atol=1e-5)

        t_fwd = benchmark(
            f"{name} jacfwd",
            lambda: full_jacobian_forward(model, x),
        )
        t_rev = benchmark(
            f"{name} jacrev",
            lambda: full_jacobian_reverse(model, x),
        )

        faster = "jacfwd" if t_fwd < t_rev else "jacrev"
        print(f"{name}: d_in={d_in}, d_out={d_out}")
        print(f"  jacfwd time per run: {t_fwd * 1e6:.2f} µs")
        print(f"  jacrev time per run: {t_rev * 1e6:.2f} µs")
        print(f"  sample ||J||:        {jac_fwd.norm().item():.6f}")
        print(f"  faster: {faster}")
        print("")

    print("Scalar loss gradient d(sum(y))/dx on the many-output case")
    scalar_model = MLP(d_in=10, d_out=1000).to(device)
    scalar_x = torch.randn(10, device=device)

    grad_fwd = scalar_grad_forward(scalar_model, scalar_x)
    grad_rev = scalar_grad_reverse(scalar_model, scalar_x)
    assert torch.allclose(grad_fwd, grad_rev, rtol=1e-4, atol=1e-5)

    t_grad_fwd = benchmark(
        "scalar grad forward",
        lambda: scalar_grad_forward(scalar_model, scalar_x),
    )
    t_grad_rev = benchmark(
        "scalar grad reverse",
        lambda: scalar_grad_reverse(scalar_model, scalar_x),
    )

    print(f"  forward-derived grad time: {t_grad_fwd * 1e6:.2f} µs")
    print(f"  reverse-mode grad time:    {t_grad_rev * 1e6:.2f} µs")
    print(f"  sample ||grad||:           {grad_rev.norm().item():.6f}")


if __name__ == "__main__":
    main()
