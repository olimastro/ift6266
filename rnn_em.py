import numpy as np
import theano
import theano.tensor as T
import sys
import matplotlib.pylab as plt
import cPickle as pkl

from blocks import initialization
from blocks.bricks import Tanh
from blocks.bricks.recurrent import LSTM
#from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model

from scipy.io.wavfile import read
from scipy.io.wavfile import write

from sklearn.mixture import GMM

np.random.seed(1234)
np.seterr(all='warn')

EXP_PATH = "/Tmp/mastropo/"

"""
    This class will train a mixture of gaussian.
    First, an RNN will produce a hidden state.
    Second, the EM algorithm will fit a GMM over the next time step
    which is the target.
    Third, the gradient will be computed with the LL of this GMM
    and the hidden state.
"""
class RNN_EM :
    def __init__(self, number_of_mix=3, learning_rate=0.01, input_dim=4800, samplerate=48000, model_saving=True, load=False) :
        self.model_saving=model_saving
        self.load = load
        self.lr = learning_rate
        self.number_of_mix = number_of_mix
        self.input_dim = input_dim
        self.samplerate = samplerate
        self.best_ll = np.inf

        self.bprop, self.fprop = self.build_theano_functions()

    # Theano will take care of everything before the EM step.
    # We have to give it the fixed param of the EM for computation of LL.
    def build_theano_functions(self) :
        #import pdb ; pdb.set_trace()
        x = T.fmatrix('x')
        s = T.fvector('s')
        mu = T.fvector('mu')
        mu = T.reshape(mu,(self.number_of_mix,1))
        pi = T.fvector('pi')

        lstm = LSTM(
            dim=self.input_dim/4,
            weights_init=IsotropicGaussian(0.5),
            biases_init=Constant(1))
        lstm.initialize()
        h, c = lstm.apply(x)
        h = h[0][0][-1]

        LL = T.sum(pi*(1./(T.sqrt(2.*np.pi)*s))*T.exp(\
            -0.5*(h-mu)**2/T.reshape(s,(self.number_of_mix,1))**2.).sum(axis=1))
        cost = -T.log(LL)

        #cg = ComputationGraph(cost)
        #self.cg = cg
        #parameters = cg.parameters
        model = Model(cost)
        self.model = model
        parameters = model.parameters

        grads = T.grad(cost, parameters)
        updates = []
        for i in range(len(grads)) :
            updates.append(tuple([parameters[i], parameters[i] - self.lr*grads[i]]))

        gradf = theano.function([x,s,mu,pi],[cost],updates=updates)
        f = theano.function([x],[h])

        return gradf, f


    def init_em_model(self, data):
        data = data.reshape((len(data),1))
        inc = self.samplerate

        gmm = GMM(self.number_of_mix, covariance_type="spherical")
        gmm.fit(data[0:inc])
        gmm.init_params = ""

        #for i in range(inc, len(data)-inc, inc):
        for i in range(inc, len(data)/1000, inc):
            gmm.fit(data[i:i+inc])

        self.gmm = gmm


    def get_gmm_param(self) :
        mus = self.gmm.means_.reshape(self.number_of_mix,)
        sigmas = self.gmm.covars_.reshape(self.number_of_mix,)
        pis = self.gmm.weights_.reshape(self.number_of_mix,)
        return np.array([sigmas, mus, pis], dtype=np.float32)


    def save_model(self, cost, not_best=False) :
        if not self.model_saving :
            return
        prefix = "best_"
        name = "rnn_em_params.pkl"
        if not_best :
            prefix = ''
            cost = -np.inf

        if cost < self.best_ll :
            self.best_ll = cost
            params = self.model.get_parameter_values()
            f = open(EXP_PATH+prefix+name,'w')
            pkl.dump(params, f)
            f.close()


    def load_model(self, best=True) :
        if best :
            prefix = "best_"
        else :
            prefix = ''
        name = "rnn_em_params.pkl"
        f = open(EXP_PATH+prefix+name)
        params = pkl.load(f)
        f.close()
        return params


    def train(self, data, epochs=50):
        if self.load :
            print "Loading previously saved model"
            self.model.set_parameter_values(self.load_model())

        data = self.prepare_data(data)
        print "Initializing GMM"
        self.init_em_model(data)

        blocks = float(np.floor(self.samplerate*60.*5.))
        costs = np.zeros((epochs,len(data)/blocks))

        for epoch in range(epochs):
            print
            print "New epoch #", epoch
            cost = 0.
            k = 1
            nan_flag = False
            for i in range(164226337, (len(data)-self.input_dim), self.input_dim) :
                #import ipdb ; ipdb.set_trace()
                sys.stdout.write('\rComputing LL on %d/%d examples'%(i, data.shape[0]))
                sys.stdout.flush()
                x = data[i:i+self.input_dim].flatten()
                y = data[i+self.input_dim:i+2*self.input_dim].flatten()
                y = y.reshape((len(y),1))
                self.gmm.fit(y)
                gmm_param_list = self.get_gmm_param()
                # Train the RNN with the likelihood over the parametrized distribution (found with EM)
                # with x_t+158M
                _cost = self.bprop(x[np.newaxis],gmm_param_list[0],gmm_param_list[1].reshape((self.number_of_mix,1)),gmm_param_list[2])
                cost += _cost[0]

                nblocks = np.floor(i/blocks)
                if nblocks >= k and nblocks <= costs.shape[1] :
                    k+=1
                    costs[epoch,nblocks-1] = cost-np.sum(costs[epoch,:nblocks-1])
                    self.save_model(cost)

                if np.isnan(cost) :
                    print
                    print "WARNING : NaN detected in cost, dumping to files and exiting"
                    nan_flag = True
                    break

            f = open(EXP_PATH+"rnn_em_cost_gpu.npy",'w')
            np.save(f, costs)
            f.close()

            if nan_flag :
                self.save_model(0, not_best=True)
                sys.exit()


    # normalize the data in [-1,1]
    # and bring it to 0 mean 1 variance
    def prepare_data(self, data) :
        data = data.astype(np.float32)
        data = data/np.max(data)
        data = (data-np.average(data))/np.std(data)
        return data



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
    print "Loading data"
    data = read("/Tmp/mastropo/XqaJ2Ol5cC4.wav")
    samplerate = data[0]
    data = data[1]

    model = RNN_EM(input_dim=2*(samplerate/10), samplerate=samplerate, model_saving=False, load=True)
    model.train(data)
