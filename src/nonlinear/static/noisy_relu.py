from .nrelu import NReLU


class NoisyReLU(NReLU):
    def __init__(self):
        super().__init__()
