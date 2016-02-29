import numpy as np
import sys
import theano
import theano.tensor as T

from blocks.bricks.recurrent import LSTM
from sklearn.mixture import GMM

np.random.seed(1234)


def fit(data) :
    gmm = GMM(2, covariance_type="spherical")
    gmm.fit(data)
    return gmm


def make_data(amount=1000000) :
    data = np.zeros((amount,2))
    p = np.random.random_sample()
    print "Using two mixtures with "+str(p)+" and "+str(1-p)
    mix1 = int(p*amount)
    data[:mix1] = np.random.normal([1,2],0.5,(mix1,2))
    data[mix1:] = np.random.normal([3,5],1.,(amount-mix1,2))

    return data


def KLDivergence(model, data)
    # lets try to compute the KL divergence on samples of the gmm
    # and the true data points
    samples = model.sample(10000)
    data = data[26:26+10000]
    

# compute prob of a data point under the gmm model
def prob_gmm(gmm, x)
    mu = gmm.means_
    pi = gmm.weights_
    si = gmm.covars_


class Model:
    def __init__(self, input_dim):
        self.lstm = LSTM(input_dim/4)

    def make_theano_function(self):
        x = T.dvector('x')

        h = self.lstm.apply(x)

        func = theano.function([x], [h[0]])

        # this comes from the likelihood computed by the gmm model from
        # the fit it did on the hidden layer of the lstm
        # with the x_t+1 points
        likelihood = T.dscalar('ll')
        ll = -T.log(likelihood)


        
    


if __name__ == "__main__" :
    data = make_data()
    np.random.shuffle(data)
    KLDivergence(fit(data), data)
