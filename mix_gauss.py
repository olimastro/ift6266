import numpy as np
import theano
import theano.tensor as T
import sys

np.random.seed(1234)

class Mixture :
    def __init__(self, learning_rate=0.001, batch_size=1) :
        self.lr = learning_rate
        self.batch_size = batch_size

        mu = np.array([1.,1.,2.,2.]).reshape(2,2)
        sigma = np.array([0.1,0.2])
        lambdas = np.array([0.5,0.5])

        self.mu = theano.shared(mu, 'mu')
        self.sigma = theano.shared(sigma, 'sigma')
        self.lambdas = theano.shared(lambdas, 'lambdas')

        self.bprop = self.make_theano_function()


    def make_theano_function(self) :
        x = T.dvector('x')

        norm = ((x-self.mu)**2).sum(axis=1, keepdims=True)

        loss = T.log(T.sum(self.lambdas*(1./(2.*np.pi*self.sigma**2))*T.exp(-0.5*norm/self.sigma)))

        gradm = T.grad(loss, self.mu)
        grads = T.grad(loss, self.sigma)
        gradl = T.grad(loss, self.lambdas)

        gradf = theano.function(
                [x],
                [loss],
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
            ll=0 

            for i in range(0, data.shape[0], self.batch_size) :
                sys.stdout.write('\rComputing LL on %d/%d examples'%(i, data.shape[0]))
                sys.stdout.flush()
                _ll = self.bprop(data[i:i+self.batch_size].flatten())
                #import pdb ; pdb.set_trace()
                if i%1000==True :
                    import pdb ; pdb.set_trace()
                    self.printt(ll)
                ll+= _ll[0]

            cst = (data.shape[0]/self.batch_size)
            ll = ll/cst

            import pdb ; pdb.set_trace()
            self.printt()
            loss[epoch] = ll

    def printt(self,ll) :
        print 
        print ll
        print self.mu.get_value()
        print self.sigma.get_value()
        print self.lambdas.get_value()


def make_data(amount=1000000) :
    data = np.zeros((amount,2))
    p = np.random.random_sample()
    print "Using two mixtures with "+str(p)+" and "+str(1-p)
    mix1 = int(p*amount)
    data[:mix1] = np.random.normal([1,2],0.5,(mix1,2))
    data[mix1:] = np.random.normal([3,5],1.,(amount-mix1,2))
    np.random.shuffle(data)

    return data


if __name__ == "__main__" :
    #import matplotlib.pylab as plt
    #data = make_data(10000)
    #plt.plot(data[:,0],data[:,1],'.')
    #plt.show()
    #sys.exit()
    mix = Mixture()
    mix.train()

