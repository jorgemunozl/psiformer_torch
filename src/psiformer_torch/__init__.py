from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["logdet_matmul", "LogDetMatmul"]

if TYPE_CHECKING:
    from .logdet_matmul import LogDetMatmul, logdet_matmul


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .logdet_matmul import LogDetMatmul, logdet_matmul

        return {"logdet_matmul": logdet_matmul, "LogDetMatmul": LogDetMatmul}[name]
    raise AttributeError(name)
