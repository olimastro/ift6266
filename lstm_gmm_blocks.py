import numpy as np
import theano
import theano.tensor as T
import sys
import cPickle as pkl

from blocks.algorithms import GradientDescent, Adam
from blocks.bricks.simple import Linear, NDimensionalSoftmax
from blocks.bricks.recurrent import LSTM
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.initialization import IsotropicGaussian, Constant, Uniform, Orthogonal
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.transformers import Mapping

from convolution import CONV
from extensions import SaveParams
from scheme import ShuffledBatchChunkScheme
from transformer import ReshapeTransformer, spec_mapping

from scipy.io.wavfile import write
from sklearn.mixture import GMM

np.random.seed(1111)
np.seterr(all='warn')

############----PARAMETERS----##############
EXP_PATH = "/Tmp/mastropo/"
DATAPATH = "/Tmp/mastropo/song.hdf5"
NAME = "lstm_gmm_2lconv"
EPOCHS = 50
EPS = 1e-7

#------------------GMM---------------------#
# x mixtures = 3*x params for time serie
NMIXTURES = 20
# [0] : time dimension for one sequence
# [1] : batch size (doesn't make sense to be more than 1?)
# [2] : input dimension (1 for time serie)
# [3] : sequence dimension
# [4] : output dimension, should be equal to number of GMM params
DIMS_TUPLE = (16000, 1, 1, 10, NMIXTURES*3)
# list[i] = dim of ith layer
#LSTM_DIM_LIST = [512, 512, 512]
LSTM_DIM_LIST = [800]

#-----------------CONV---------------------#
WITH_CONV = False
# list of convolutional sequence parameters
# list[i] := params for ith layer
# list[i][0:3] := params for conv layer
#   [0] = (,) filter size, [1] = nb of filters per channel, [2] = nb channels
# list[i][3] := pooling size
CONV_PARAMS = [[(4,4), 64, 1, (3,3)], [(3,3), 32, 64, (2,2)]]
#PARAMS = [[(4,4), 50, 1, (3,3)]]
IMAGE_SIZE = (30, 257) #now a dummy param as the img size will be infered in get_datafile method
############################################

"""
    This class will train a mixture of gaussian according to Alex Graves' paper
    on generating sequence using RNN and GMM.
    In short, we will use the output of an RNN to parametrize a Gaussian Mixture
    Model. Each Gaussians will then be sampled from to generate each time points for each sequence.
    Meaning the the time dimension in the RNN is actually the sequence dimension.
    We will give to the RNN a sequence (self.sequence_dim) of time points (self.time_dim).

    (Optionnally), it can also be fed features from the spectrogram learned by a conv_net.
"""
class LSTM_GMM :
    def __init__(self, dims_tuple, lstm_dim_list, gmm_dim, learning_rate=0.0000001, samplerate=48000, with_conv=False) :
        self.debug = 0
        self.lr = learning_rate # this is useless as we use Adam
        self.orth_scale = 0.9
        self.samplerate = samplerate

        self.time_dim = dims_tuple[0]
        self.batch_dim = dims_tuple[1]
        self.input_dim = dims_tuple[2]
        self.sequence_dim = dims_tuple[3]
        self.output_dim = dims_tuple[4]
        self.gmm_dim = gmm_dim

        self.lstm_layers_dim = lstm_dim_list

        assert self.gmm_dim*3 == self.output_dim

        if with_conv :
            self.image_size = IMAGE_SIZE
        else :
            self.image_size = None


    def init_conv(self) :
        self.conv = CONV(CONV_PARAMS, self.image_size)


    def build_theano_functions(self):
        x = T.fmatrix('time_sequence')
        x = x.reshape((self.batch_dim, self.sequence_dim, self.time_dim))

        y = x[:,1:self.sequence_dim,:]
        x = x[:,:self.sequence_dim-1,:]

        # if we try to include the spectrogram features
        spec_dims = 0
        if self.image_size is not None :
            print "Convolution activated"
            self.init_conv()
            spec = T.ftensor4('spectrogram')
            spec_features, spec_dims = self.conv.build_conv_layers(spec)
            print "Conv final dims =", spec_dims
            spec_dims = np.prod(spec_dims)
            spec_features = spec_features.reshape(
                (self.batch_dim, self.sequence_dim-1, spec_dims))
            x = T.concatenate([x, spec_features], axis=2)

        layers_input = [x]
        dims =np.array([self.time_dim + spec_dims])
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
                            biases_init=Constant(0),
                            name="linear"+str(layer))
            linear.initialize()
            lstm_input = linear.apply(layers_input[layer])

            # the lstm wants batch X sequence X time
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

        # this is where Alex Graves' paper starts
        print "Last linear transform dim :", dims[1:].sum()
        output_transform = Linear(dims[1:].sum(),
                                  self.output_dim,
                                  weights_init=Orthogonal(self.orth_scale),
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
        pis = T.reshape(
                    T.nnet.softmax(
                        T.reshape(y_hat[:,:,:self.gmm_dim], ((self.sequence_dim-1)*self.batch_dim, self.gmm_dim))),
                    (self.batch_dim, (self.sequence_dim-1), self.gmm_dim))
        sig = T.exp(y_hat[:,:,self.gmm_dim:self.gmm_dim*2])+1e-6
        mus = y_hat[:,:,self.gmm_dim*2:]

        pis = pis[:,:,:,np.newaxis]
        mus = mus[:,:,:,np.newaxis]
        sig = sig[:,:,:,np.newaxis]
        y = y[:,:,np.newaxis,:]

        y = T.patternbroadcast(y, (False, False, True, False))
        mus = T.patternbroadcast(mus, (False, False, False, True))
        sig = T.patternbroadcast(sig, (False, False, False, True))

        # sum likelihood with targets
        # see blog for this crazy Pr() = sum log sum prod
        # axes :: (batch, sequence, mixture, time)
        expo_term = -0.5*((y-mus)**2)/sig**2
        coeff = T.log(T.maximum(1./(T.sqrt(2.*np.pi)*sig), EPS))
        #coeff = T.log(1./(T.sqrt(2.*np.pi)*sig))
        sequences = coeff + expo_term
        log_sequences = T.log(pis + EPS) + T.sum(sequences, axis=3, keepdims=True)

        log_sequences_max = T.max(log_sequences, axis=2, keepdims=True)

        LL = -(log_sequences_max + T.log(EPS + T.sum(T.exp(log_sequences - log_sequences_max), axis=2, keepdims=True))).mean()
        LL.name = "summed_likelihood"

        model = Model(LL)
        self.model = model
        parameters = model.parameters

        algorithm = GradientDescent(
            cost=LL,
            parameters=model.parameters,
            step_rule=Adam())

        f = theano.function([x],[pis, sig, mus])

        return algorithm, f


    def train(self):
        print "Loading data"
        datafile = self.get_datafile()
        nbexamples = datafile.num_examples
        nbexamples -= nbexamples%(self.sequence_dim*self.time_dim)

        train_stream = ReshapeTransformer(
            DataStream(
                dataset=datafile,
                iteration_scheme=ShuffledBatchChunkScheme(
                    nbexamples, self.sequence_dim*self.time_dim)),
            self.sequence_dim,
            self.time_dim)

        if self.image_size is not None :
            train_stream = Mapping(train_stream, spec_mapping, add_sources=['spectrogram'])

        print "Building Theano Graph"
        algorithm, self.fprop = self.build_theano_functions()

        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=train_stream,
            model=self.model,
            extensions=[
                FinishAfter(after_n_epochs=EPOCHS),
                TrainingDataMonitoring(
                    [aggregation.mean(self.model.outputs[0])],
                    prefix="train",
                    after_epoch=True),
                Printing(),
                SaveParams(EXP_PATH+NAME, after_epoch=True)
            ])

        main_loop.run()


    def load_model(self) :
        model_path = EXP_PATH+NAME+"_params.pkl"
        print "Loading model at", model_path
        f = open(model_path)
        params = pkl.load(f)
        f.close()
        algorithm, self.fprop = self.build_theano_functions()
        self.model.set_parameter_values(params)


    def generate(self, seed=None, minutes=0.5):
        print "Generating module"
        timestep = self.time_dim*(self.sequence_dim-1)
        samples = minutes*self.samplerate*60
        song = np.zeros(samples, dtype=np.float32)

        if seed is None :
            datafile = self.get_datafile()
            seed = datafile.get_data(None, range(timestep))
            seed = seed[0].flatten()

        song[:timestep] = seed

        print
        for i in range(0, len(song)-self.time_dim-timestep, self.time_dim) :
            sys.stdout.write('\rGenerating %d/%d samples'%(i, samples))
            sys.stdout.flush()

            params = self.fprop(song[i:i+timestep].reshape(
                (self.batch_dim, self.sequence_dim-1, self.time_dim)))
            try :
                song[i+timestep:i+timestep+self.time_dim] = self.sample_from_gmm(params)
            except ValueError :
                import ipdb ; ipdb.set_trace()

        write(EXP_PATH+"generation.wav", self.samplerate, song)


    def sample_from_gmm(self, params) :
        # There is one set of mixture param for every timestep
        # remember the shape is [batch, sequence, mixture, time]
        pis = np.array(params[0])
        sig = np.array(params[1])
        mus = np.array(params[2])

        gmm = GMM(self.gmm_dim, covariance_type='spherical', init_params='')
        gmm.weights_ = pis[0,-1,:]
        gmm.means_ = mus[0,-1,:]
        gmm.covars_= sig[0,-1,:]

        return gmm.sample(self.time_dim).flatten()


    def get_datafile(self) :
        try :
            datafile = H5PYDataset(DATAPATH, which_sets=('train', ),
                                   sources=['time_sequence'], load_in_memory=True)
        except IOError :
            print "Could not find the hdf5 file. Will try to generate it"
            raise NotImplementedError

        if self.image_size is not None :
            print "Image size attribute is not None, need to infer the image size of the spectrogram"
            # temporarly create all the streams and stuff to make one mapping, inside the
            # mapping are the image size. Probably a cleaner way to do this.
            nbexamples = datafile.num_examples
            nbexamples -= nbexamples%(self.sequence_dim*self.time_dim)
            dummy_stream = ReshapeTransformer(
                DataStream(
                    dataset=datafile,
                    iteration_scheme=ShuffledBatchChunkScheme(
                        nbexamples, self.sequence_dim*self.time_dim)),
                self.sequence_dim,
                self.time_dim)
            dummy_stream = Mapping(dummy_stream, spec_mapping, add_sources=['spectrogram'])
            dummy_epoch_iterator = dummy_stream.get_epoch_iterator()
            dummy_data = next(dummy_epoch_iterator)
            dummy_data = dummy_data[1]
            self.image_size = (dummy_data.shape[2], dummy_data.shape[3])
            print "Img size found, it should be =", self.image_size

            del nbexamples
            del dummy_stream
            del dummy_epoch_iterator
            del dummy_data

        return datafile



if __name__ == "__main__" :
    model = LSTM_GMM(DIMS_TUPLE, LSTM_DIM_LIST, NMIXTURES, samplerate=16000, with_conv=WITH_CONV)
    model.train()
    model.load_model()
    model.generate()
