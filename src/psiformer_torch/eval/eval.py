import logging
import matplotlib.pyplot as plt
import torch
from psiformer import PsiFormer
from config import Model_Config, CHECKPOINT_PATH


def use_checkpoint():
    loaded = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
    model = PsiFormer(Model_Config())
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    else:
        # Backwards compatibility for checkpoints saved as raw state_dict
        state_dict = loaded

    model.load_state_dict(state_dict)
    model.eval()
    y = torch.linspace(0, 1, 100)
    x = torch.stack(
        [torch.tensor([_, 0.0, 0.0]) for _ in y],
        dim=0
    )
    probability = torch.exp(model(x))**2
    log_space = model(x)
    plt.plot(y, probability.detach().numpy(), label="probability")
    plt.plot(y, log_space.detach().numpy(), label="model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(nam)s %(levelname)s: %(message)s"
                        )
    logger = logging.getLogger("Beginning")
    logger.info("Starting")

    use_checkpoint()
