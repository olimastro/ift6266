import numpy as np
import theano
import theano.tensor as T
import sys
import cPickle as pkl

from blocks.bricks.simple import Linear, NDimensionalSoftmax
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant, Uniform, Orthogonal
from blocks.model import Model

from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy import signal

from sklearn.mixture import GMM

np.random.seed(1111)
np.seterr(all='warn')

EXP_PATH = "/Tmp/mastropo/"
NAME = "lstm_fourier"
EPOCHS = 50
WINDOW_SIZE = 512
OVERLAP = WINDOW_SIZE/2
# [0] : sequence of points in time
# [1] : batch size (doesn't make sense to be more than 1?)
# [2] : input dimension, the spectrogram has window_size+2 (DC comp for amplitude and phase)
# [3] : output dimension, we have 2 gaussians of two param per freq bin
DIMS_LIST = [16, 1, WINDOW_SIZE+2, (WINDOW_SIZE+2)*2]
# list[i] = dim of ith layer
#LSTM_DIM_LIST = [512, 512]
LSTM_DIM_LIST = [1200]

"""
    This class will train a mixture of gaussian.
    First, an RNN will produce a hidden state.
    Second, the EM algorithm will fit a GMM over the next time step
    which is the target.
    Third, the gradient will be computed with the LL of this GMM
    and the hidden state.
"""
class LSTM_FOURIER :
    def __init__(self, dims_list, lstm_dim_list, learning_rate=0.00001, samplerate=48000, model_saving=False, load=False) :
        self.debug = 0
        self.model_saving=model_saving
        self.load = load
        self.lr = learning_rate
        #self.lr = 0
        self.samplerate = samplerate
        self.best_ll = np.inf

        self.time_dim = dims_list[0]
        self.batch_dim = dims_list[1]
        self.input_dim = dims_list[2]
        self.output_dim = dims_list[3]

        self.lstm_layers_dim = lstm_dim_list


    def build_theano_functions(self) :
        x = T.ftensor3('x') # shape of input : batch X time X value
        y = T.ftensor3('y')
        z = T.ftensor3('z')

        layers_input = [x]
        dims =np.array([self.input_dim])
        for dim in self.lstm_layers_dim :
            dims = np.append(dims, dim)
        print "Dimensions =", dims

        # layer is just an index of the layer
        for layer in range(len(self.lstm_layers_dim)) :

            # before the cell, input, forget and output gates, x needs to
            # be transformed
            linear = Linear(dims[layer],
                            dims[layer+1]*4,
                            #weights_init=Uniform(mean=data_mean, std=1),
                            weights_init=IsotropicGaussian(mean=1.,std=1),
                            biases_init=Constant(0),
                            name="linear"+str(layer))
            linear.initialize()
            lstm_input = linear.apply(layers_input[layer])

            # the lstm wants batch X time X value
            lstm = LSTM(
                dim=dims[layer+1],
                weights_init=IsotropicGaussian(mean=0.,std=0.5),
                biases_init=Constant(1),
                name="lstm"+str(layer))
            lstm.initialize()
            # hack to use Orthogonal on lstm w_state
            lstm.W_state.set_value(Orthogonal().generate(np.random, lstm.W_state.get_value().shape))
            h, _dummy = lstm.apply(lstm_input)

            layers_input.append(h)

        # the idea is to have one gaussian parametrize every frequency bin
        print "Last linear transform dim :", dims[1:].sum()
        output_transform = Linear(dims[1:].sum(),
                                  self.output_dim,
                                  weights_init=IsotropicGaussian(mean=0., std=1),
                                  biases_init=Constant(0),
                                  #use_bias=False,
                                  name="output_transform")
        output_transform.initialize()
        if len(self.lstm_layers_dim) == 1 :
            print "hallo there, only one layer speaking"
            y_hat = output_transform.apply(layers_input[-1])
        else :
            y_hat = output_transform.apply(T.concatenate(layers_input[1:], axis=2))

        sig = T.nnet.relu(y_hat[:,:,:self.output_dim/2])+0.05
        mus = y_hat[:,:,self.output_dim/2:]

        # sum likelihood with targets
        # sum inside log accross mixtures, sum outside log accross time
        inside_expo = -0.5*((y-mus)**2)/sig**2
        expo = T.exp(inside_expo)
        coeff = 1./(T.sqrt(2.*np.pi)*sig)
        inside_log = T.log(coeff*expo)
        inside_log_max = T.max(inside_log, axis=2, keepdims=True)
        LL = -(inside_log_max + T.log(T.sum(T.exp(inside_log - inside_log_max), axis=2, keepdims=True))).sum()

        #zinside_expo = -0.5*((z-mus)**2)/sig**2
        #zexpo = T.exp(zinside_expo)
        #zcoeff = pis*(1./(T.sqrt(2.*np.pi)*sig))
        #zinside_log = (zcoeff*zexpo).sum(axis=2)
        #zLL = -(T.log(zinside_log)).sum()

        model = Model(LL)
        self.model = model
        parameters = model.parameters

        grads = T.grad(LL, parameters)
        updates = []
        lr = T.scalar('lr')
        for i in range(len(grads)) :
            #updates.append(tuple([parameters[i], parameters[i] - self.lr*grads[i]]))
            updates.append(tuple([parameters[i], parameters[i] - lr*grads[i]]))

        #gradf = theano.function([x, y],[LL],updates=updates, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
        if self.debug :
            gradf = theano.function([x, y, lr],[LL, mus, sig],updates=updates)
        else :
            #gradf = theano.function([x, y, z],[zLL],updates=updates)
            gradf = theano.function([x, y, lr],[LL],updates=updates)
        f = theano.function([x],[sig, mus])

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
        print "Loading model at", EXP_PATH+prefix+name
        f = open(EXP_PATH+prefix+name)
        params = pkl.load(f)
        f.close()
        self.bprop, self.fprop = self.build_theano_functions(0.,0.)
        self.model.set_parameter_values(params)


    def train(self, data, epochs=EPOCHS):
        #import ipdb ; ipdb.set_trace()
        print "Creating the spectrogram"
        # Spec is shape (time, features)
        # first half of features is the freq's amplitudes
        # second half of features is the freq's phases
        spec  = self.prepare_data(data.astype(np.float32))
        print "Building Theano Graph"
        self.bprop, self.fprop = self.build_theano_functions()

        if self.load :
            self.load_model()

        blocks = int(spec.shape[0]/(float(len(data))/(self.samplerate*60)))

        for epoch in range(epochs):
            print
            print "New epoch #", epoch
            cost = 0.
            k = 1
            nan_flag = False
            for i in range(0, (len(spec)-2*self.time_dim), self.time_dim) :
                sys.stdout.write('\rComputing LL on %d/%d examples'%(i, spec.shape[0]))
                sys.stdout.flush()

                x = spec[i:i+self.time_dim]
                x = x.reshape((self.batch_dim, self.time_dim, self.input_dim))

                #y = data[i+self.time_dim:i+2*self.time_dim]
                y = spec[i+1:i+self.time_dim+1]
                y = y.reshape((self.batch_dim, self.time_dim, self.input_dim))

                #z = data[self.samplerate*3600:self.samplerate*3600+self.time_dim]
                #z = z.reshape((self.batch_dim, self.time_dim, self.input_dim))
                #z_ = z[:,:,np.newaxis,:]

                if self.debug :
                    l = self.bprop(x, y, self.lr)
                    #make_nice_print(l)
                    #import ipdb ; ipdb.set_trace()
                else :
                    #cost = self.bprop(z,z_,z_)
                    cost = self.bprop(x, y, self.lr)

                nblocks = np.floor(i/blocks)
                if nblocks >= k :
                    print cost
                    #make_nice_print(l)
                    k+=1
                    #costs[epoch,nblocks-1] = cost-np.sum(costs[epoch,:nblocks-1])
                    self.save_model(cost[0])

                if np.isnan(cost) :
                    print
                    print "WARNING : NaN detected in cost, dumping to files and exiting"
                    nan_flag = True
                    break

            if False and epoch%3 :
                begin = self.samplerate*60*5
                self.generate(epoch, data[begin:begin+self.time_dim].reshape((self.batch_dim, self.time_dim, self.input_dim)))

            if (epoch+2)%5 :
                self.lr*=10

            #f = open(EXP_PATH+NAME+"_cost.npy",'w')
            #np.save(f, costs)
            #f.close()

            if nan_flag :
                self.save_model(0, not_best=True)
                sys.exit()


    def generate(self, epoch, begin, minutes=1):
        import pdb ; pdb.set_trace()
        samples = minutes*self.samplerate*60
        true_len = int(np.floor(samples/self.time_dim)*self.time_dim)
        song = np.zeros(true_len, dtype=np.float32)

        print
        for i in range(self.time_dim, true_len-self.time_dim) :
            sys.stdout.write('\rGenerating %d/%d samples'%(i, samples))
            sys.stdout.flush()

            params = self.fprop(song[i:i+self.time_dim].reshape(
                (self.batch_dim, self.time_dim, self.input_dim)))
            song[i+1] = self.sample_from_gmm(params)

        #song *= 2**30
        #song = song.astype(np.int32)
        write(EXP_PATH+"generation"+str(epoch)+".wav", self.samplerate, song)


    def sample_from_gmm(self, params) :
        # There is one set of mixture param for every timestep
        # remember the shape is [batch, time, mixture, value]
        pis = np.array(params[0])
        sig = np.array(params[1])
        mus = np.array(params[2])

        #sequence = np.empty(self.time_dim, dtype=np.float32)
        #for i in range(self.time_dim) :
        gmm = GMM(self.gmm_dim, covariance_type='spherical', init_params='')
        gmm.weights_ = pis[0,0,:]
        gmm.means_ = mus[0,0,:]
        gmm.covars_= sig[0,0,:]

            #sequence[i] = gmm.sample()

        return gmm.sample()


    def prepare_data(self, data, regenerate=False) :
        data = data[self.samplerate*60*2:len(data)-self.samplerate*60*2]
        window = signal.cosine(WINDOW_SIZE)

        f_data = signal.spectrogram(data, window=window, nperseg=WINDOW_SIZE, noverlap=OVERLAP, mode='complex')
        f_data = np.swapaxes(f_data[2], 0, 1)
        real = np.real(f_data)
        rmax = np.max(real) if np.absolute(np.min(real)) < np.max(real) else np.absolute(np.min(real))
        real /= rmax
        cplx = np.imag(f_data)
        cmax = np.max(cplx) if np.absolute(np.min(cplx)) < np.max(cplx) else np.absolute(np.min(cplx))
        cplx /= cmax

        return np.append(real, cplx, axis=1)



def make_nice_print(l) :
    print
    print "Cost =", l[0]
    print "pis =", l[1][0,2]
    print "mus =", l[2][0,2]
    print "sig =", l[3][0,2]


if __name__ == "__main__" :
    print "Loading data"
    data = read("/Tmp/mastropo/XqaJ2Ol5cC4.wav")
    samplerate = data[0]
    data = data[1]

    model = LSTM_FOURIER(DIMS_LIST, LSTM_DIM_LIST, samplerate=samplerate)
    model.train(data)
    #model.load_model()
    #model.generate(1,0)
