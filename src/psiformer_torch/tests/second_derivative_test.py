"""
Test of determinant first and second derivative using implementation from
backwards.py
"""
import torch
from psiformer_torch.backwards import DetWithCofactor


def f1(X: torch.Tensor) -> torch.Tensor:
    return X*2


def function(A: torch.Tensor) -> torch.Tensor:
    return torch.det(A)


def my_function(A: torch.Tensor):
    # A = f1(A)
    return DetWithCofactor.apply(A)


def first_derivative(func, x):
    """
    Test for the first derivative
    """
    if x.grad is not None:
        x.grad.zero_()
    Y = func(x)
    Y.backward()
    print("Derivate of Y wrt X:\n", x.grad)


def second_derivative(func, x):
    """
    Test for the second derivative
    """
    return torch.autograd.functional.hessian(lambda t: func(t), x)


def eval_second(func, x):
    x = x.clone().requires_grad_()
    hess = second_derivative(func, x)
    print("Hession", hess)


if __name__ == "__main__":

    # If X is a singular function then the derivative becomes zero.
    # If matriz singular then it doesn't try to compute the gradients
    X_singular = torch.tensor([[1.0, 2.0],
                              [2.0, 4.0]], requires_grad=True)

    X_n_singular = torch.tensor([[3.0, 2.0],
                                [2.0, 2.0]], requires_grad=True)

    print("Using my Tweaked Function (second derivative)")
    eval_second(my_function, X_n_singular)
    eval_second(my_function, X_singular)

    print("Using torch built in (second derivative)")
    eval_second(function, X_n_singular)
    eval_second(function, X_singular)

    # dot = make_dot(Y, params={"X": X})
    # dot.render("det_trace_graph", format="png", cleanup=True)
