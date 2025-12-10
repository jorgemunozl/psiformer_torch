import torch
from torch.autograd import Function


class MySquare(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # ctx is not 'self', it's just a context object PyTorch gives you
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return 1 + x * grad_output


x = torch.tensor([2.0], requires_grad=True)
y = MySquare.apply(x)

y.backward()

print(x.grad)
