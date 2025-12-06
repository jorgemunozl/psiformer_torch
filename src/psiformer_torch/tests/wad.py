import random
import wandb
import time

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="alvaro18ml-university-of-minnesota",
    # Set the wandb project where this run will be logged.
    project="Psiformer",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 100
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = epoch
    loss = epoch**2
    time.sleep(10)
    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()
