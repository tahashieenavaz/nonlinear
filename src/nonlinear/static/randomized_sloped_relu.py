import random
from .sloped_relu import SlopedReLU


class RandomizedSlopedReLU(SlopedReLU):
    def __init__(self):
        super().__init__(alpha=random.uniform(1, 10))
