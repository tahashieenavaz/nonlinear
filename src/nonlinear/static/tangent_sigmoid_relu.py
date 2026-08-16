from .tsrelu import TSReLU


class TangentSigmoidReLU(TSReLU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
