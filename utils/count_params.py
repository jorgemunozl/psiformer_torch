#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.SimpleNamespace(init=lambda *args, **kwargs: None)

from psiformer_torch.config import (  # noqa: E402
    PSIFORMER_TORCH_LARGE_MODEL,
    PSIFORMER_TORCH_SMALL_MODEL,
)


def count_params_from_model(config) -> tuple[int, int]:
    from psiformer_torch.psiformer import PsiFormer

    model = PsiFormer(config)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_params_formula(config) -> int:
    n_embd = config.n_embd
    n_layer = config.n_layer
    n_features = config.n_features
    n_det = config.n_determinants
    spin_sum = config.n_spin_up + config.n_spin_down

    l0 = n_embd * (n_features + 1) + n_embd

    layer = 12 * n_embd * n_embd + 13 * n_embd
    layers_total = n_layer * layer

    envelope = 2 * n_det * spin_sum
    orbital_linear = n_det * spin_sum * n_embd + n_det * spin_sum
    det_logits = n_det
    orbital_head = envelope + orbital_linear + det_logits

    jastrow = 2

    return l0 + layers_total + orbital_head + jastrow


def main() -> None:
    models = {
        "small": PSIFORMER_TORCH_SMALL_MODEL,
        "large": PSIFORMER_TORCH_LARGE_MODEL,
    }

    for name, config in models.items():
        try:
            print("Counting parameters from model")
            total, trainable = count_params_from_model(config)
            source = "torch"
        except ModuleNotFoundError as e:
            # print the error
            print("Error: ", e)
            print("Counting parameters from formula")
            total = count_params_formula(config)
            trainable = total
            source = "formula"

        print(
            f"{name}: total={total:,} trainable={trainable:,} source={source}"
        )


if __name__ == "__main__":
    main()
