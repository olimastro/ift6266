import numpy as np
import theano
import theano.tensor as T
import sys
import matplotlib.pylab as plt

np.random.seed(1234)
np.seterr(all='warn')

"""
    This class is doing gradient descent in order to find the parameters
    of a mixture of gaussian.
"""
class Mixture :
    def __init__(self, learning_rate=0.01, batch_size=1) :
        self.lr = learning_rate
        self.batch_size = batch_size

        mu = np.array([1.,1.,2.,2.]).reshape(2,2)
        sigma = np.array([0.2,0.5])
        lambdas = np.array([0.3,0.7])

        self.mu = theano.shared(mu, 'mu')
        self.sigma = theano.shared(sigma, 'sigma')
        self.lambdas = theano.shared(lambdas, 'lambdas')

        self.bprop = self.make_theano_function()


    def make_theano_function(self) :
        x = T.dvector('x')

        norm = ((x-self.mu)**2).sum(axis=1, keepdims=True)
        lambdas = T.nnet.softmax(self.lambdas)
        #sigmas = T.nnet.relu(self.sigma)+0.0001

        loss = -T.log(T.sum(lambdas*(1./(2.*np.pi*self.sigma**2))*T.exp(-0.5*norm/self.sigma**2)))
        #loss = T.sum(lambdas*(1./(2.*np.pi*sigmas**2))*T.exp(-0.5*norm/sigmas))
        #loss = T.sum(lambdas*T.exp(-0.5*norm/sigmas))

        gradm = T.grad(loss, self.mu)
        grads = T.grad(loss, self.sigma)
        gradl = T.grad(loss, self.lambdas)

        gradf = theano.function(
                [x],
                [loss, gradm, grads, gradl],
                updates = [
                    (self.mu, self.mu - self.lr*gradm),
                    (self.sigma, self.sigma - self.lr*grads),
                    (self.lambdas, self.lambdas - self.lr*gradl),
                    ]
                )
        return gradf


    def train(self, epochs=1) :
        loss = np.zeros(epochs)
        data = make_data()

        for epoch in range(epochs) :
            print "Computing epoch #", epoch
            np.random.shuffle(data)
            ll=0
            gradient = np.zeros(data.shape[0])

            for i in range(0, data.shape[0], self.batch_size) :
                sys.stdout.write('\rComputing LL on %d/%d examples'%(i, data.shape[0]))
                sys.stdout.flush()
                _ll, gm, gs, gl = self.bprop(data[i:i+self.batch_size].flatten())
                #import pdb ; pdb.set_trace()
                #self.printt(ll)
                #if i%1000==True :
                #    #self.printt(ll, [gm,gs,gl])
                #    self.printt(ll)
                #    #import pdb ; pdb.set_trace()
                gradient[i] = np.linalg.norm(gs)
                ll+= _ll

            plt.plot(gradient)
            plt.show()
            cst = (data.shape[0]/self.batch_size)
            ll = ll/cst

            #import pdb ; pdb.set_trace()
            self.printt(ll)
            loss[epoch] = ll

    def printt(self,ll,gradient_list=None) :
        print
        print ll
        print self.mu.get_value()
        print self.sigma.get_value()
        print softmax(self.lambdas.get_value())
        if not gradient_list is None :
            for g in gradient_list :
                print g


def softmax(x) :
    y = np.exp(x)
    y = y/np.sum(y)
    return y

def make_data(amount=1000000) :
    data = np.zeros((amount,2))
    p = np.random.random_sample()
    print "Using two mixtures with "+str(p)+" and "+str(1-p)
    mix1 = int(p*amount)
    data[:mix1] = np.random.normal([1,2],0.5,(mix1,2))
    data[mix1:] = np.random.normal([3,5],1.,(amount-mix1,2))

    return data


if __name__ == "__main__" :
    #import matplotlib.pylab as plt
    #data = make_data(10000)
    #plt.plot(data[:,0],data[:,1],'.')
    #plt.show()
    #sys.exit()
    mix = Mixture()
    mix.train(epochs=50)

