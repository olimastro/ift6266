import numpy as np
import theano
import theano.tensor as T
import sys
import matplotlib.pylab as plt
import cPickle as pkl

from blocks.bricks.simple import Linear, NDimensionalSoftmax
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant, Uniform, Orthogonal
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
NMIXTURES = 2
# [0] : sequence of points in time
# [1] : batch size (doesn't make sense to be more than 1?)
# [2] : input dimension (1 for time serie)
# [3] : lstm dimension, hidden representation
# [4] : output dimension, should be equal to number of GMM params
DIMS_TUPLE = (8000, 1, 1, 100, NMIXTURES*3)

"""
    This class will train a mixture of gaussian.
    First, an RNN will produce a hidden state.
    Second, the EM algorithm will fit a GMM over the next time step
    which is the target.
    Third, the gradient will be computed with the LL of this GMM
    and the hidden state.
"""
class LSTM_GMM :
    def __init__(self, dims_tuple, gmm_dim, learning_rate=0.001, samplerate=48000, model_saving=True, load=False) :
        self.debug = False
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
                       weights_init=Uniform(mean=data_mean, std=1),
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
            weights_init=IsotropicGaussian(mean=0.,std=0.5),
            biases_init=Constant(1))
        lstm.initialize()
        # hack to use Orthogonal on lstm w_state
        lstm.W_state.set_value(Orthogonal().generate(np.random, lstm.W_state.get_value().shape))
        h, _dummy = lstm.apply(lstm_input)

        # this is where Alex Graves' paper starts
        output_transform = Linear(self.lstm_dim,
                                  self.output_dim,
                                  #weights_init=Uniform(mean=data_mean, std=data_std),
                                  weights_init=IsotropicGaussian(mean=0., std=1),
                                  use_bias=False,
                                  name="output_transform")
        output_transform.initialize()
        y_hat = output_transform.apply(h)

        # transforms to find each gmm params (mu, pi, sig)
        # small hack to softmax a 3D tensor
        pis = T.reshape(
                    T.nnet.softmax(
                        T.nnet.sigmoid(
                            T.reshape(y_hat[:,:,0:self.gmm_dim], (self.time_dim*self.batch_dim, self.gmm_dim)))),
                    (self.batch_dim, self.time_dim, self.gmm_dim))
        #sig = T.exp(y_hat[:,:,self.gmm_dim:self.gmm_dim*2])
        sig = T.nnet.relu(y_hat[:,:,self.gmm_dim:self.gmm_dim*2])+0.1
        mus = 10.*T.tanh(y_hat[:,:,self.gmm_dim*2:])
        #mus = y_hat[:,:,self.gmm_dim*2:]

        pis = pis[:,:,:,np.newaxis]
        mus = mus[:,:,:,np.newaxis]
        sig = sig[:,:,:,np.newaxis]
        y = y[:,:,np.newaxis,:]

        #pis=theano.printing.Print()(pis)
        #mus=theano.printing.Print()(mus)
        #sig=theano.printing.Print()(sig)

        # sum likelihood with targets
        # sum inside log accross mixtures, sum outside log accross time
        #LL = -T.log((pis*(1./(T.sqrt(2.*np.pi)*sig))*T.exp(-0.5*((y-mus)**2)/sig**2)).sum(axis=2)).sum()
        inside_expo = -0.5*((y-mus)**2)/sig**2
        #inside_expo=theano.printing.Print()(inside_expo)
        #expo = T.exp(-0.5*((y-mus)**2)/sig**2)
        expo = T.exp(inside_expo)
        #expo=theano.printing.Print()(expo)
        coeff = pis*(1./(T.sqrt(2.*np.pi)*sig))
        #coeff=theano.printing.Print()(coeff)
        inside_log = (coeff*expo).sum(axis=2)
        #inside_log=theano.printing.Print()(inside_log)
        LL = -(T.log(inside_log)).sum()

        model = Model(LL)
        self.model = model
        parameters = model.parameters

        grads = T.grad(LL, parameters)
        updates = []
        for i in range(len(grads)) :
            updates.append(tuple([parameters[i], parameters[i] - self.lr*grads[i]]))

        #gradf = theano.function([x, y],[LL],updates=updates, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
        if self.debug :
            gradf = theano.function([x, y],[LL, pis, mus, sig],updates=updates)
        else :
            gradf = theano.function([x, y],[LL],updates=updates)
        f = theano.function([x],[pis, sig, mus])

        return gradf, f


    def save_model(self, cost, not_best=False) :
        if not self.model_saving :
            return
        prefix = "best_"
        name = NAME+"_params.pkl"
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
        name = NAME+"_params.pkl"
        f = open(EXP_PATH+prefix+name)
        params = pkl.load(f)
        f.close()
        return params


    def train(self, data, epochs=20):
        if self.load :
            print "Loading previously saved model"
            self.model.set_parameter_values(self.load_model())

        data, data_mean, data_std  = self.prepare_data(data)
        print data_mean, data_std
        print "Building Theano Graph"
        self.bprop, self.fprop = self.build_theano_functions(data_mean, data_std)

        blocks = float(np.floor(self.samplerate*60.*1.))
        costs = np.zeros((epochs,len(data)/blocks))

        for epoch in range(epochs):
            print
            print "New epoch #", epoch
            cost = 0.
            k = 1
            nan_flag = False
            for i in range(0, (len(data)-2*self.time_dim), self.time_dim) :
                sys.stdout.write('\rComputing LL on %d/%d examples'%(i, data.shape[0]))
                sys.stdout.flush()

                x = data[i:i+self.time_dim]
                x = x.reshape((self.batch_dim, self.time_dim, self.input_dim))
                y = data[i+self.time_dim:i+2*self.time_dim]
                y = y.reshape((self.batch_dim, self.time_dim, self.input_dim))
                y = y[:,:,np.newaxis,:]

                if self.debug :
                    make_nice_print(self.bprop(x,y))
                    import ipdb ; ipdb.set_trace()
                else :
                    cost = self.bprop(x,y)

                nblocks = np.floor(i/blocks)
                if nblocks >= k and nblocks <= costs.shape[1] :
                    print cost
                    k+=1
                    costs[epoch,nblocks-1] = cost-np.sum(costs[epoch,:nblocks-1])
                    self.save_model(cost)

                if np.isnan(cost) :
                    print
                    print "WARNING : NaN detected in cost, dumping to files and exiting"
                    nan_flag = True
                    break

            if epoch%2 :
                begin = self.samplerate*60*5
                self.generate(epoch, data[begin:begin+self.time_dim].reshape((self.batch_dim, self.time_dim, self.input_dim)))

            f = open(EXP_PATH+NAME+"_cost.npy",'w')
            np.save(f, costs)
            f.close()

            if nan_flag :
                self.save_model(0, not_best=True)
                sys.exit()


    def generate(self, epoch, begin, minutes=2):
        samples = minutes*self.samplerate*60
        true_len = int(np.floor(samples/self.time_dim)*self.time_dim)
        song = np.empty(true_len, dtype=np.float32)

        params = self.fprop(begin)
        song[:self.time_dim] = self.sample_from_gmm(params)
        print
        for i in range(self.time_dim, true_len-self.time_dim, self.time_dim) :
            sys.stdout.write('\rGenerating %d/%d samples'%(i, samples))
            sys.stdout.flush()

            params = self.fprop(song[i-self.time_dim:i].reshape(
                (self.batch_dim, self.time_dim, self.input_dim)))
            song[i:i+self.time_dim] = self.sample_from_gmm(params)

        song *= 2**30
        song = song.astype(np.int32)
        write(EXP_PATH+"generation"+str(epoch)+".wav", self.samplerate, song)


    def sample_from_gmm(self, params) :
        # There is one set of mixture param for every timestep
        # remember the shape is [batch, time, mixture, value]
        pis = np.array(params[0])
        sig = np.array(params[1])
        mus = np.array(params[2])

        sequence = np.empty(self.time_dim, dtype=np.float32)
        for i in range(self.time_dim) :
            gmm = GMM(self.gmm_dim, covariance_type='spherical', init_params='')
            gmm.weights_ = pis[0,i,:]
            gmm.means_ = mus[0,i,:]
            gmm.covars_= sig[0,i,:]

            sequence[i] = gmm.sample()

        return sequence



    # normalize the data in [-1,1]
    # and bring it to 0 mean 1 variance <-- could be a bad idea...
    def prepare_data(self, data) :
        data = data.astype(np.float32)
        data = data/np.max(data)
        #data = (data-np.average(data))/np.std(data)
        return data, np.mean(data), np.std(data)



def make_nice_print(l) :
    print
    print "Cost =", l[0]
    print "pis =", l[1][0,2]
    print "mus =", l[2][0,2]
    print "sig =", l[3][0,2]


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

    model = LSTM_GMM(DIMS_TUPLE, NMIXTURES, samplerate=samplerate)
    model.train(data)
