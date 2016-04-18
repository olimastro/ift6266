import numpy as np
from scipy import signal
from fuel.transformers import SourcewiseTransformer

WINDOW_SIZE = 512
OVERLAP = 512/2

class ReshapeTransformer(SourcewiseTransformer):
    def __init__(self, data_stream, shape1, shape2, **kwargs):
        self.shape1 = shape1
        self.shape2 = shape2
        super(ReshapeTransformer, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)


    def transform_source_example(self, source_example, _):
        raise NotImplementedError("This should not happend")


    def transform_source_batch(self, source_batch, _):
        return np.array(source_batch.reshape((self.shape1, self.shape2)))


def spec_mapping(wave) :
    wave = wave[0]
    window = signal.cosine(WINDOW_SIZE)

    spec = signal.spectrogram(wave, window=window, nperseg=WINDOW_SIZE, noverlap=OVERLAP, mode='psd')
    spec = np.swapaxes(spec[2], 2, 1)
    spec = spec[np.newaxis,:,:,:]

    return (spec,)
