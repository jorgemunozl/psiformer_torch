class PsiformerConfig:
    def __init__(self, model_size="base",
                 learning_rate=1e-4,
                 batch_size=32):
        self.model_size = model_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

