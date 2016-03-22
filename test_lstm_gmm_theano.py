import numpy as np
import theano
import theano.tensor as T
import sys
import matplotlib.pylab as plt
import cPickle as pkl

from blocks.bricks.simple import Linear, NDimensionalSoftmax
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.model import Model

from scipy.io.wavfile import read
from scipy.io.wavfile import write

from sklearn.mixture import GMM
from theano.compile.nanguardmode import NanGuardMode
np.random.seed(1234)
np.seterr(all='warn')

EXP_PATH = "/Tmp/mastropo/"
NAME = "lstm_gmm"
# x mixtures = 3*x params for time serie
NMIXTURES = 3
# [0] : sequence of points in time
# [1] : batch size (doesn't make sense to be more than 1?)
# [2] : input dimension (1 for time serie)
# [3] : lstm dimension, hidden representation
# [4] : output dimension, should be equal to number of GMM params
DIMS_TUPLE = (10, 1, 1, 5, NMIXTURES*3)

"""
    This class is a copy of lstm_gmm and is used as testing stuff
"""
class LSTM_GMM :
    def __init__(self, dims_tuple, gmm_dim, learning_rate=0.001, samplerate=48000, model_saving=False, load=False) :
        self.model_saving=model_saving
        self.load = load
        self.lr = learning_rate
        self.samplerate = samplerate
        self.best_ll = np.inf

        self.time_dim = dims_tuple[0]
        self.batch_dim = dims_tuple[1]
        self.input_dim = dims_tuple[2]
        self.lstm_dim = dims_tuple[3]
        self.output_dim = dims_tuple[4]
        self.gmm_dim = gmm_dim

        assert self.gmm_dim*3 == self.output_dim


    def build_theano_functions(self, data_mean, data_std) :
        x = T.ftensor3('x') # shape of input : batch X time X value
        y = T.ftensor3('y')

        # before the cell, input, forget and output gates, x needs to
        # be transformed
        linear_transforms = []
        for transform in ['c','i','f','o'] :
            linear_transforms.append(
                Linear(self.input_dim,
                       self.lstm_dim,
                       weights_init=Uniform(mean=data_mean, std=data_std),
                       #weights_init=IsotropicGaussian(mean=1.,std=1),
                       biases_init=Constant(data_mean),
                       name=transform+"_transform")
            )

        for transform in linear_transforms :
            transform.initialize()

        linear_applications = []
        for transform in linear_transforms :
            linear_applications.append(
                transform.apply(x))

        lstm_input = T.concatenate(linear_applications, axis=2)

        # the lstm wants batch X time X value
        lstm = LSTM(
            dim=self.lstm_dim,
            weights_init=IsotropicGaussian(mean=0.5,std=1),
            biases_init=Constant(1))
        lstm.initialize()
        h, _dummy = lstm.apply(lstm_input)

        # this is where Alex Graves' paper starts
        output_transform = Linear(self.lstm_dim,
                                  self.output_dim,
                                  #weights_init=Uniform(mean=data_mean, std=data_std),
                                  weights_init=IsotropicGaussian(mean=0., std=1),
                                  biases_init=Constant(1),
                                  name="output_transform")
        output_transform.initialize()
        y_hat = output_transform.apply(h)

        # transforms to find each gmm params (mu, pi, sig)
        #pis = NDimensionalSoftmax.apply(y_hat[:,:,0:self.gmm_dim])
        # small hack to softmax a 3D tensor
        pis = T.reshape(
                    T.nnet.softmax(
                        T.reshape(y_hat[:,:,0:self.gmm_dim], (self.time_dim*self.batch_dim, self.gmm_dim)))
                    , (self.batch_dim, self.time_dim, self.gmm_dim))
        #sig = T.exp(y_hat[:,:,self.gmm_dim:self.gmm_dim*2])
        sig = T.nnet.relu(y_hat[:,:,self.gmm_dim:self.gmm_dim*2])+0.1
        mus = y_hat[:,:,self.gmm_dim*2:]

        pis = pis[:,:,:,np.newaxis]
        mus = mus[:,:,:,np.newaxis]
        sig = sig[:,:,:,np.newaxis]
        y = y[:,:,np.newaxis,:]

        #sig=theano.printing.Print()(sig)

        # sum likelihood with targets
        # sum inside log accross mixtures, sum outside log accross time
        #LL = -T.log((pis*(1./(T.sqrt(2.*np.pi)*sig))*T.exp(-0.5*((y-mus)**2)/sig**2)).sum(axis=2)).sum()
        expo = T.exp(-0.5*((y-mus)**2)/sig**2)
        test_expo = theano.function([x,y],[expo, mus, sig])
        return test_expo

        coeff = pis*(1./(T.sqrt(2.*np.pi)*sig))
        inside_log = (coeff*expo).sum(axis=2)
        LL = -(T.log(inside_log)).sum()


        model = Model(LL)
        self.model = model
        parameters = model.parameters

        grads = T.grad(LL, parameters)
        updates = []
        for i in range(len(grads)) :
            updates.append(tuple([parameters[i], parameters[i] - self.lr*grads[i]]))

        #gradf = theano.function([x, y],[LL],updates=updates, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
        gradf = theano.function([x, y],[LL],updates=updates)
        f = theano.function([x],[pis, sig, mus])

        return gradf, f


    def test(self, data) :
        data, data_mean, data_std  = self.prepare_data(data)
        print "Building Theano Graph"
        func = self.build_theano_functions(data_mean, data_std)

        x = data[0:10]
        x = x.reshape((self.batch_dim, self.time_dim, self.input_dim))
        y = data[11900:11910]
        y = y.reshape((self.batch_dim, self.time_dim, self.input_dim))
        y = y[:,:,np.newaxis,:]

        expo, mus, sig = func(x,y)
        import ipdb ; ipdb.set_trace()


    def prepare_data(self, data) :
        data = data.astype(np.float32)
        data = data/np.max(data)
        #data = (data-np.average(data))/np.std(data)
        return data, np.mean(data), np.std(data)


if __name__ == "__main__" :
    print "Loading data"
    data = read("/Tmp/mastropo/XqaJ2Ol5cC4.wav")
    samplerate = data[0]
    data = data[1]

    model = LSTM_GMM(DIMS_TUPLE, NMIXTURES, samplerate=samplerate)
    model.test(data)
