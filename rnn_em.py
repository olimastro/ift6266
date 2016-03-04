import numpy as np
import theano
import theano.tensor as T
import sys
import matplotlib.pylab as plt

from blocks.bricks import MLP, Tanh
from blocks.algorithms import GradientDescent, Scale
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.extensions import FinishAfter, Timing
from blocks.main_loop import MainLoop
from blocks.model import Model

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from sklearn.mixture import GMM

np.random.seed(1234)
np.seterr(all='warn')

"""
    This class will train a mixture of gaussian
    on top of an RNN in a back and forth fashion.
"""
class RNN_EM :
    def __init__(self, number_of_mix=3, learning_rate=0.01, input_dim=4800):
        self.lr = learning_rate
        self.bprop, self.fprop = self.build_theano_functions()
        self.number_of_mix = number_of_mix
        self.input_dim = input_dim

        self.extensions = [Timing(), FinishAfter(after_n_epochs=1)]


    # Theano will take care of everything before the EM step.
    # We have to give it the fixed param of the EM for computation of LL.
    def build_theano_functions(self) :
        x = T.dvector('x')
        y = T.dvector('y')
        s = T.dvector('s')
        mu = T.dvector('mu')
        mu = T.reshape(mu,(3,1))
        pi = T.dvector('pi')

        mlp = MLP([Tanh()], [4800,4800],
                  weights_init=IsotropicGaussian(0.01),
                  biases_init=Constant(0))
        mlp.initialize()
        h = mlp.apply(x)

        LL = T.sum(pi*(1./(T.sqrt(2.*np.pi)*s))*(\
            -0.5*(y-mu)**2/T.reshape(s,(3,1))**2.).sum(axis=1))
        cost = -T.log(LL)

        cg = ComputationGraph(cost)
        parameters = cg.parameters
        grads = T.grad(cost, parameters)
        updates = []
        for i in range(len(grads)) :
            updates.append(tuple([parameters[i], parameters[i] - self.lr*grads[i]]))

        gradf = theano.function([x,y,s,mu,pi],[h,LL],updates=updates)
        f = theano.function([x],[h])

        return gradf, f


    def init_em_model(self, data):
        inc = 48000

        sigmas = np.zeros((len(data)/inc, self.number_of_mix))
        mus = np.zeros((len(data)/inc, self.number_of_mix))
        pis = np.zeros((len(data)/inc, self.number_of_mix))

        gmm = GMM(self.number_of_mix, covariance_type="spherical")
        gmm.fit(data[0:inc])

        mus[0] = gmm.means_.reshape(self.number_of_mix,)
        sigmas[0] = gmm.covars_.reshape(self.number_of_mix,)
        pis[0] = gmm.weights_.reshape(self.number_of_mix,)
        gmm.init_params = ""

        for i in range(inc, len(data)-inc, inc):
            gmm.fit(data[i:i+inc])

            mus[i/inc] = gmm.means_.reshape(self.number_of_mix,)
            sigmas[i/inc] = gmm.covars_.reshape(self.number_of_mix,)
            pis[i/inc] = gmm.weights_.reshape(self.number_of_mix,)

        self.gmm = gmm


    def get_gmm_param(self) :
        mus = self.gmm.means_.reshape(self.number_of_mix,)
        sigmas = self.gmm.covars_.reshape(self.number_of_mix,)
        pis = self.gmm.weights_.reshape(self.number_of_mix,)
        return [sigmas, mus, pis]


    def train(self, data, epochs=1):
        np.random.shuffle(data)
        self.init_em_model(data)

        for epoch in range(epochs):
            for i in (0, len(data)-self.input_dim, self.input_dim) :
                x = data[i:i+self.input_dim].flatten()
                h = self.fprop(x)
                # The RNN should have produced the next distribution over x_t
                self.gmm.fit(np.array(h).reshape((self.input_dim,1)))
                gmm_param_list = self.get_gmm_param()
                # Train the RNN with the likelihood over the parametrized distribution (found with EM)
                # with x_t+1
                y = data[i+self.input_dim:i+2*self.input_dim].flatten()
                self.bprop(x,y,gmm_param_list[0],gmm_param_list[1].reshape((3,1)),gmm_param_list[2])




def make_data(amount=1000000):
    data = np.zeros((amount, 1))
    p1 = 0.3
    p2 = 0.2
    p3 = 0.5
    print "Using three mixtures with "+str(p1)+" and "+str(p2)+" and "+str(p3)
    mix1 = int(p1*amount)
    mix2 = int(p2*amount)
    mix3 = amount-mix1-mix2
    data[:mix1] = np.random.normal(-1,0.5,(mix1,1))
    data[mix1:mix1+mix2] = np.random.normal(2,1.,(mix2,1))
    data[mix1+mix2:] = np.random.normal(5,1.5,(mix3,1))

    return data


if __name__ == "__main__" :
    model = RNN_EM()
    model.train(make_data())
