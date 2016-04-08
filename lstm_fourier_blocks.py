import numpy as np
import theano
import theano.tensor as T
import sys
import cPickle as pkl

from blocks.algorithms import GradientDescent, AdaGrad
from blocks.bricks.simple import Linear, NDimensionalSoftmax
from blocks.bricks.recurrent import LSTM
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.initialization import IsotropicGaussian, Constant, Uniform, Orthogonal
from blocks.main_loop import MainLoop
from blocks.model import Model

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from scheme import OverlapSequentialScheme

from sklearn.mixture import GMM

np.random.seed(1111)
np.seterr(all='warn')

EXP_PATH = "/Tmp/mastropo/"
DATAPATH = "/Tmp/mastropo/fouried_song.hdf5"
NAME = "lstm_fourier"
EPOCHS = 50
WINDOW_SIZE = 512
OVERLAP = WINDOW_SIZE/2
# [0] : sequence of points in time
# [1] : batch size (doesn't make sense to be more than 1?)
# [2] : input dimension, the spectrogram has window_size+2 (DC comp for amplitude and phase)
# [3] : output dimension, we have 2 gaussians of two param per freq bin
DIMS_LIST = [32, 1, WINDOW_SIZE+2, (WINDOW_SIZE+2)*2]
# list[i] = dim of ith layer
LSTM_DIM_LIST = [600, 600]
#LSTM_DIM_LIST = [1200]

"""
    This class will train a mixture of gaussian.
    First, an RNN will produce a hidden state.
    Second, the EM algorithm will fit a GMM over the next time step
    which is the target.
    Third, the gradient will be computed with the LL of this GMM
    and the hidden state.
"""
class LSTM_FOURIER :
    def __init__(self, dims_list, lstm_dim_list, samplerate=48000, model_saving=True, load=False) :
        self.debug = 0
        self.model_saving=model_saving
        self.load = load
        #self.lr = 0
        self.orth_scale = 0.9
        self.samplerate = samplerate
        self.best_ll = np.inf

        self.time_dim = dims_list[0]
        self.batch_dim = dims_list[1]
        self.input_dim = dims_list[2]
        self.output_dim = dims_list[3]

        self.lstm_layers_dim = lstm_dim_list


    def build_theano_functions(self) :
        # shape of theano inpu is time+1 X features
        x = T.fmatrix('frequency_sequence')
        x = x.reshape((self.batch_dim, self.time_dim+1, self.input_dim))

        y = x[:,1:self.time_dim+1,:]
        x = x[:,:self.time_dim,:]

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
                            weights_init=Orthogonal(self.orth_scale),
                            #weights_init=IsotropicGaussian(mean=1.,std=1),
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
            lstm.W_state.set_value(
                self.orth_scale*Orthogonal().generate(np.random, lstm.W_state.get_value().shape))
            h, _dummy = lstm.apply(lstm_input)

            layers_input.append(h)

        # the idea is to have one gaussian parametrize every frequency bin
        print "Last linear transform dim :", dims[1:].sum()
        output_transform = Linear(dims[1:].sum(),
                                  self.output_dim,
                                  #weights_init=IsotropicGaussian(mean=0., std=1),
                                  weights_init=Orthogonal(self.orth_scale),
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
        LL.name = "summed_likelihood"

        model = Model(LL)
        self.model = model

        algorithm = GradientDescent(
            cost=LL,
            parameters=model.parameters,
            step_rule=AdaGrad())

        f = theano.function([x],[sig, mus])

        return algorithm, f


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


    def train(self):
        print "Loading data"
        datafile = self.get_datafile()
        nbexamples = datafile.num_examples

        train_stream = DataStream(
            dataset=datafile,
            iteration_scheme=OverlapSequentialScheme(
                nbexamples, self.time_dim))

        print "Building Theano Graph"
        algorithm, self.fprop = self.build_theano_functions()

        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=train_stream,
            extensions=[
                FinishAfter(after_n_epochs=EPOCHS),
                TrainingDataMonitoring(
                    [self.model.outputs[0]],
                    prefix="train",
                    after_epoch=True,
                    every_n_batches=4000),
                #ProgressBar(),
                Printing()
            ])

        main_loop.run()


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


    def get_datafile(self) :
        datafile = H5PYDataset(DATAPATH, which_sets=('train', ),
                               sources=['frequency_sequence'], load_in_memory=False)
        return datafile


    def prepare_data(self, data, regenerate=False) :
        data = data[self.samplerate*60*1:len(data)-self.samplerate*60*1]
        window = signal.cosine(WINDOW_SIZE)

        f_data = signal.spectrogram(data, window=window, nperseg=WINDOW_SIZE, noverlap=OVERLAP, mode='complex')
        f_data = np.swapaxes(f_data[2], 0, 1)
        real = np.real(f_data)
        real /= np.max(real)
        cplx = np.imag(f_data)
        cplx /= np.max(cplx)

        return np.append(real, cplx, axis=1)



if __name__ == "__main__" :
    model = LSTM_FOURIER(DIMS_LIST, LSTM_DIM_LIST, samplerate=16000)
    model.train()
    #model.load_model()
    #model.generate(1,0)
