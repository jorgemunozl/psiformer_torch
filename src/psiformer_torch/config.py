class PsiformerConfig:
    def __init__(self,
                 model_size="base",
                 nelectrons=10,
                 slater_dets=5,
                 hidden_units=128,
                 dim=64,
                 residual_blocks=4,
                 jastrow_blocks=3,
                 learning_rate=1e-4,
                 batch_size=32
                 ):
        self.nelectrons = nelectrons
        self.model_size = model_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
