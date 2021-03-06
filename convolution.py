import numpy as np
import theano
import theano.tensor as T

from blocks.bricks.conv import ConvolutionalSequence, MaxPooling, Convolutional
from blocks.bricks.simple import Rectifier
from blocks.initialization import IsotropicGaussian, Constant

#-----------These params are for the debugging------------#
# list of convolutional sequence parameters
# list[i] := params for ith layer
# list[i][0:3] := params for conv layer
#   [0] = (,) filter size, [1] = nb of filters per channel, [2] = nb channels
# list[i][3] := pooling size
PARAMS = [[(4,4), 200, 1, (3,3)], [(3,3), 100, 200, (2,2)]]
IMAGE_SIZE = (100,100)
#---------------------------------------------------------#

"""
    This class implements a CNN using the ConvolutionalSequence brick.
    IT IS NOT meant to be used by itself. It builds the theano computation
    graph up to the output of the CNN which is returned by the only
    method here.
"""

class CONV :
    def __init__(self, params, image_size, with_flatten=True) :
        self.params = params
        self.layers = len(self.params)
        self.image_size = image_size
        self.with_flatten = with_flatten


    def build_conv_layers(self, image=None) :

        if image is None :
            image = T.ftensor4('spectrogram')
        else :
            image = image

        conv_list = []
        for layer in range(self.layers) :
            layer_param = self.params[layer]
            conv_layer = Convolutional(layer_param[0], layer_param[1], layer_param[2])
            pool_layer = MaxPooling(layer_param[3])

            conv_layer.name = "convolution"+str(layer)
            pool_layer.name = "maxpooling"+str(layer)

            conv_list.append(conv_layer)
            conv_list.append(pool_layer)
            conv_list.append(Rectifier())

        conv_seq = ConvolutionalSequence(
            conv_list,
            self.params[0][2],
            image_size=self.image_size,
            weights_init=IsotropicGaussian(std=0.5, mean=0),
            biases_init=Constant(0))

        conv_seq._push_allocation_config()
        conv_seq.initialize()
        out = conv_seq.apply(image)

        return out, conv_seq.get_dim('output')


# DEBUG, not tested to be working by itself
if __name__ == '__main__' :
    conv = CONV(PARAMS, IMAGE_SIZE)
    import ipdb ; ipdb.set_trace()
    image, out = conv.build_conv_layers()
    f = theano.function([image], [out])
    img = np.random.random(100*100).astype(np.float32)
    img = img.reshape((1, 1, 100, 100))
    features = f(img)
