import numpy as np
import matplotlib.pylab as plt
import theano
import theano.tensor as T

class MLP :
    def __init__(self, input_dim, hidden_dim=12, output_dim=2, learning_rate=0.01, batch_size=1) :
        self.input_dim = input_dim
        self.hidden_dim= hidden_dim
        self.output_dim= output_dim
        self.lr = learning_rate
        self.batch_size = batch_size

        w1 = self.init_weights(self.hidden_dim, self.input_dim)
        w2 = self.init_weights(self.output_dim, self.hidden_dim)
        b1 = np.zeros(self.hidden_dim, dtype=w1[0].dtype)
        b2 = np.zeros(self.output_dim, dtype=w1[0].dtype)

        self.w1 = theano.shared(w1, 'w1')
        self.w2 = theano.shared(w2, 'w2')
        self.b1 = theano.shared(b1, 'b1')
        self.b2 = theano.shared(b2, 'b2')

        self.fprop, self.bprop = self.make_theano_functions()


    # make theano function
    def make_theano_functions(self) :
        x  = T.dmatrix('x')
        h1 = T.dot(x, self.w1.T) + self.b1
        a1 = 1. / (1. + T.exp(-h1))
        h2 = T.dot(a1,self.w2.T) + self.b2
        a2 = T.nnet.softmax(h2)
        
        f = theano.function([x], a2)

        y  = T.dmatrix('y')
        loss = T.mean(T.sum(y*-T.log(a2), axis=1))

        gradw1 = T.grad(loss, self.w1)
        gradw2 = T.grad(loss, self.w2)
        gradb1 = T.grad(loss, self.b1)
        gradb2 = T.grad(loss, self.b2)

        gradf = theano.function(
                [x, y],
                [loss, a2],
                updates = [
                    (self.w1, self.w1-self.lr*gradw1),
                    (self.w2, self.w2-self.lr*gradw2),
                    (self.b1, self.b1-self.lr*gradb1),
                    (self.b2, self.b2-self.lr*gradb2)
                    ]
                )

        return f, gradf


    #takes the two dimension of the weight matrix
    # lines := number of units of this layer
    # columns := number of dimensions of input
    def init_weights(self, x, y) :
        interval = 1./np.sqrt(y)
        W = np.random.uniform(low=-interval,high=interval,size=(x,y))
        return W


    def train_model(self, train_data, valid_data, test_data, epochs=2) :
        loss = np.zeros(epochs)
        missclass = np.zeros(epochs)

        for epoch in range(epochs) :
            ll = 0 ; mc = 0 ;
            for i in range(0, train_data[0].shape[0], self.batch_size) :
                _ll, _mc = self.train(w, train_data[1][i:i+self.batch_size])
                ll += _ll
                mc += _mc
            loss[epoch] = ll/(train_data[0].shape[0]/self.batch_size)
            missclass[epoch] = mc

        #import pdb ; pdb.set_trace()
        plt.plot(missclass)
        #plt.plot(loss)
        plt.show()


    def train(self, data, labels) :
        targets = onehot(labels, self.output_dim)
        loss, class_probs = self.bprop(data, targets)
        predictions = np.argmax(class_probs, axis=1)
        missclass = np.sum(predictions-labels != 0)
        #import pdb ; pdb.set_trace()
        return loss, missclass



def onehot(x, nb_class) :
    y = np.zeros((x.shape[0],nb_class))
    for i in range(y.shape[0]) :
        y[i][x[i]-1] = 1
    return y


if __name__ == '__main__' :
    import cPickle as pkl
    import gzip

    f=gzip.open('/u/mastropo/hw/hw_A15/mnist.pkl.gz')
    mnist_data=pkl.load(f)

    mnist_train = mnist_data[0]
    mnist_valid = mnist_data[1]
    mnist_test = mnist_data[2]

    #import pdb ; pdb.set_trace()
    mlp = MLP(len(mnist_train[0][0])/2+1, 12, 10, batch_size=100)
    mlp.train_model(mnist_train, mnist_valid, mnist_test, epochs=50)
