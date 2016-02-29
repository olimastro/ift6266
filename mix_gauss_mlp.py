import numpy as np
import theano
import theano.tensor as T
import sys

np.random.seed(1234)

class Mixture_MLP :
    def __init__(self, indim, hdim, batch_size=1) :
        self.batch_size = batch_size
        self.lr = 0.001

        w_lambda = self.init_weights(2, indim)
        w_mu = self.init_weights(4, indim)
        w_sigma = self.init_weights(2, indim)
        bias_lambda = np.zeros(2)
        bias_mu = np.zeros(4)
        bias_sigma = np.zeros(2)

        print w_lambda.shape
        print w_mu.shape
        print w_sigma.shape
        print bias_lambda.shape 
        print bias_mu.shape
        print bias_sigma.shape

        self.w_lambda = theano.shared(w_lambda, 'wl')
        self.w_mu = theano.shared(w_mu, 'wm')
        self.w_sigma = theano.shared(w_sigma, 'ws')
        self.bias_lambda = theano.shared(bias_lambda, 'bl')
        self.bias_mu = theano.shared(bias_mu, 'bm')
        self.bias_sigma = theano.shared(bias_sigma, 'bs')

        self.bprop = self.make_theano_function()


    def make_theano_function(self) :
        x = T.dvector('x')
        # probability of mixtures
        lambdas = T.nnet.softmax(T.dot(x, self.w_lambda.T) + self.bias_lambda)
        lambdas = T.reshape(lambdas,(2,1))
        # averages of gaussians
        mus = T.tanh(T.dot(x, self.w_mu.T) + self.bias_mu)
        mus = T.reshape(mus,(2,2))
        # sigmas of gaussians
        sigmas = T.nnet.relu(T.dot(x, self.w_sigma.T) + self.bias_sigma)+0.000001
        sigmas = T.reshape(sigmas,(2,1))

        norm = ((x-mus)**2).sum(axis=1, keepdims=True)

        loss = T.sum(lambdas*(1./(np.pi*sigmas**2))*T.exp(-0.5*norm/sigmas))
    
        #mix1 = lambdas[0]*(1./(np.pi*sigmas[0]**2))*T.exp(-0.5*T.dot(x-mus[0],x-mus[0])/sigmas[0])
        #mix2 = lambdas[1]*(1./(np.pi*sigmas[1]**2))*T.exp(-0.5*T.dot(x-mus[1],x-mus[1])/sigmas[1])

        #loss = mix1+mix2

        gradwl = T.grad(loss, self.w_lambda)
        gradbl = T.grad(loss, self.bias_lambda)
        gradwm = T.grad(loss, self.w_mu)
        gradbm = T.grad(loss, self.bias_mu)
        gradws = T.grad(loss, self.w_sigma)
        gradbs = T.grad(loss, self.bias_sigma)

        gradf = theano.function(
                [x],
                [loss, lambdas, mus, sigmas],
                updates = [
                    (self.w_lambda, self.w_lambda-self.lr*gradwl),
                    (self.bias_lambda, self.bias_lambda-self.lr*gradbl),
                    (self.w_mu, self.w_mu-self.lr*gradwm),
                    (self.bias_mu, self.bias_mu-self.lr*gradbm),
                    (self.w_sigma, self.w_sigma-self.lr*gradws),
                    (self.bias_sigma, self.bias_sigma-self.lr*gradbs)
                    ]
                )

        return gradf
               

    def init_weights(self, x, y) :
        interval = 1./np.sqrt(y)
        W = np.random.uniform(low=-interval,high=interval,size=(x,y))
        return W


    def train(self, epochs=1) :
        loss = np.zeros(epochs)
        data = make_data()

        for epoch in range(epochs) :
            ll=0 
            l = np.zeros((2,1))
            mu = np.zeros((2,2))
            s = np.zeros((2,1))

            for i in range(0, data.shape[0], self.batch_size) :
                sys.stdout.write('\rComputing LL on %d/%d examples'%(i, data.shape[0]))
                sys.stdout.flush()
                _ll, _l, _m, _s = self.bprop(data[i:i+self.batch_size].flatten())
                ll+= _ll 
                l += +l
                mu += _m
                s += _s

            cst = (data.shape[0]/self.batch_size)
            l = l/cst
            mu = mu/cst
            s = s/cst
            ll = ll/cst

            print
            print l
            print mu
            print s
            print ll
            loss[epoch] = ll
    

def make_data(amount=1000000) :
    data = np.zeros((amount,2))
    p = np.random.random_sample()
    print "Using two mixtures with "+str(p)+" and "+str(1-p)
    mix1 = int(p*amount)
    data[:mix1] = np.random.normal([1,2],1.,(mix1,2))
    data[mix1:] = np.random.normal([3,5],1.5,(amount-mix1,2))
    np.random.shuffle(data)

    return data


if __name__ == "__main__" :
    mix = Mixture_MLP(2, 12)
    mix.train()

