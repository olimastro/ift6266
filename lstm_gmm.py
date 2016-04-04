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
np.random.seed(1111)
np.seterr(all='warn')

EXP_PATH = "/Tmp/mastropo/"
NAME = "lstm_gmm"
EPOCHS = 15
# x mixtures = 3*x params for time serie
NMIXTURES = 20
# [0] : sequence of points in time
# [1] : batch size (doesn't make sense to be more than 1?)
# [2] : input dimension (1 for time serie)
# [3] : lstm dimension, hidden representation
# [4] : output dimension, should be equal to number of GMM params
DIMS_TUPLE = (800, 1, 1, 800, NMIXTURES*3)
# list[i] = dim of ith layer
LSTM_DIM_LIST = [512, 512, 512]

"""
    This class will train a mixture of gaussian.
    First, an RNN will produce a hidden state.
    Second, the EM algorithm will fit a GMM over the next time step
    which is the target.
    Third, the gradient will be computed with the LL of this GMM
    and the hidden state.
"""
class LSTM_GMM :
    def __init__(self, dims_tuple, lstm_dim_list, gmm_dim, learning_rate=0.000001, samplerate=48000, model_saving=False, load=False) :
        self.debug = 1
        self.model_saving=model_saving
        self.load = load
        self.lr = learning_rate
        #self.lr = 0
        self.samplerate = samplerate
        self.best_ll = np.inf

        self.time_dim = dims_tuple[0]
        #self.time_dim = 2
        self.batch_dim = dims_tuple[1]
        self.input_dim = dims_tuple[2]
        self.output_dim = dims_tuple[4]
        self.gmm_dim = gmm_dim

        self.lstm_layers_dim = lstm_dim_list

        assert self.gmm_dim*3 == self.output_dim


    def build_theano_functions(self, data_mean, data_std) :
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
                            weights_init=Uniform(mean=data_mean, std=1),
                            #weights_init=IsotropicGaussian(mean=1.,std=1),
                            biases_init=Constant(0),
                            name="linear"+str(layer))
            linear.initialize()
            lstm_input = linear.apply(layers_input[layer])

            #linear_transforms = []
            #for transform in ['c','i','f','o'] :
            #    transform = transform+str(layer)
            #    linear_transforms.append(
            #        Linear(self.input_dim,
            #               self.lstm_layers[1][layer],
            #               weights_init=Uniform(mean=data_mean, std=1),
            #               #weights_init=IsotropicGaussian(mean=1.,std=1),
            #               biases_init=Constant(0),
            #               name=transform+"_transform")
            #    )

            #for transform in linear_transforms :
            #    transform.initialize()

            #linear_applications = []
            #for transform in linear_transforms :
            #    linear_applications.append(
            #        transform.apply(layers_input[layer]))

            #lstm_input = T.concatenate(linear_applications, axis=2)

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

        # this is where Alex Graves' paper starts
        print "Last linear transform dim :", dims[1:].sum()
        output_transform = Linear(dims[1:].sum(),
                                  self.output_dim,
                                  #weights_init=Uniform(mean=data_mean, std=data_std),
                                  weights_init=IsotropicGaussian(mean=0., std=1),
                                  use_bias=False,
                                  name="output_transform")
        output_transform.initialize()
        if len(self.lstm_layers_dim) == 1 :
            print "hallo there, only one layer speaking"
            y_hat = output_transform.apply(layers_input[-1])
        else :
            y_hat = output_transform.apply(T.concatenate(layers_input[1:], axis=2))

        # transforms to find each gmm params (mu, pi, sig)
        # small hack to softmax a 3D tensor
        #pis = T.reshape(
        #            T.nnet.softmax(
        #                T.nnet.sigmoid(
        #                    T.reshape(y_hat[:,:,0:self.gmm_dim], (self.time_dim*self.batch_dim, self.gmm_dim)))),
        #            (self.batch_dim, self.time_dim, self.gmm_dim))
        pis = T.reshape(
                    T.nnet.softmax(
                        T.reshape(y_hat[:,:,0:self.gmm_dim], (self.time_dim*self.batch_dim, self.gmm_dim))),
                    (self.batch_dim, self.time_dim, self.gmm_dim))
        #sig = T.exp(y_hat[:,:,self.gmm_dim:self.gmm_dim*2])
        sig = T.nnet.relu(y_hat[:,:,self.gmm_dim:self.gmm_dim*2])+0.05
        #mus = 1.5*T.tanh(y_hat[:,:,self.gmm_dim*2:])
        mus = y_hat[:,:,self.gmm_dim*2:]

        pis = pis[:,:,:,np.newaxis]
        mus = mus[:,:,:,np.newaxis]
        sig = sig[:,:,:,np.newaxis]
        y = y[:,:,np.newaxis,:]
        z = z[:,:,np.newaxis,:]

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
        inside_log = T.log(coeff*expo)
        inside_log_max = T.max(inside_log, axis=2, keepdims=True)
        #inside_log=theano.printing.Print()(inside_log)
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
            gradf = theano.function([x, y, lr],[LL, pis, mus, sig],updates=updates)
        else :
            #gradf = theano.function([x, y, z],[zLL],updates=updates)
            gradf = theano.function([x, y, lr],[LL],updates=updates)
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
        print "Loading model at", EXP_PATH+prefix+name
        f = open(EXP_PATH+prefix+name)
        params = pkl.load(f)
        f.close()
        self.bprop, self.fprop = self.build_theano_functions(0.,0.)
        self.model.set_parameter_values(params)


    def train(self, data, epochs=EPOCHS):
        #import ipdb ; ipdb.set_trace()
        data, data_mean, data_std  = self.prepare_data(data)
        print data_mean, data_std
        print "Building Theano Graph"
        self.bprop, self.fprop = self.build_theano_functions(data_mean, data_std)

        if self.load :
            self.load_model()

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

                #y = data[i+self.time_dim:i+2*self.time_dim]
                y = data[i+1:i+self.time_dim+1]
                y = y.reshape((self.batch_dim, self.time_dim, self.input_dim))
                y = y[:,:,np.newaxis,:]

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
                if nblocks >= k and nblocks <= costs.shape[1] :
                    #print cost
                    make_nice_print(l)
                    k+=1
                    #costs[epoch,nblocks-1] = cost-np.sum(costs[epoch,:nblocks-1])
                    #self.save_model(cost[0])

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



    # normalize the data in [-1,1]
    # and bring it to 0 mean 1 variance <-- could be a bad idea...
    def prepare_data(self, data) :
        data = data[self.samplerate*60*2:len(data)-self.samplerate*60*2]
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

    model = LSTM_GMM(DIMS_TUPLE, LSTM_DIM_LIST, NMIXTURES, samplerate=samplerate)
    model.train(data)
    #model.load_model()
    #model.generate(1,0)