import numpy as np
import copy
from scipy.special import expit as sigmoid
from scipy.optimize import fmin_l_bfgs_b
from matplotlib import pyplot as plt


class NeuralNetwork():

    def __init__(self, layers, functions, derivatives):
        """
        :param layers sequence describing how many neurons in each layer,
                e.g. layers = [2,10,10,5] means there are four layers: 2
                neurons in input, 10 neurons in first hidden layer, 10 neurons
                neurons in input, 10 neurons in first hidden layer, 10 neurons
                in second hidden layer, 5 in output
        :param functions - sequence of functions definig activations between layers
    :param derivatives - sequence of derivatives of fu` `   nctions provided in functions parametr,
                Note: derivatives should be computed in terms of previous function computations:
                E.g.: if function is f(x) arguments for derivative_f would be f(x):
        """
        self.parameters = []
        self.functions = functions
        self.derivatives = derivatives
        for i in range(len(layers)-1):
            from_ = layers[i]
            to_ = layers[i+1]
            #appedning W, and b for each layer
            self.parameters.append([np.random.rand(to_,from_), np.random.rand(to_)])
            x,y = self.parameters[-1]

        self.num_layers = len(layers)-1

    def forward(self, X, with_intermediate = False):
        outputs = []
        for i,(W,b) in enumerate(self.parameters):
            z_previous = None
            if len(outputs) == 0:
                z_previous = X
            else:
                z_previous = outputs[-1][-1]
            layer_a = W.dot(z_previous) + b[None,:].T
            layer_z = self.functions[i](layer_a)
            outputs.append([layer_a,layer_z])

        if with_intermediate:
            return outputs
        return outputs[-1][-1]

    def cost(self,X,Y,regularizer):
        output = self.forward(X)
        s = 0
        for l in range(self.num_layers):
            s += np.power(self.parameters[l][0],2).sum()
        return -(np.log(output)*Y).sum() + s*regularizer*0.5

    def backprop(self,X,Y, regularizer, gradient_check = False):
        outputs = self.forward(X, with_intermediate=True)
        deltas = []

        for l in reversed(range(self.num_layers)):
            if l == self.num_layers-1:
                current_delta = -Y + outputs[-1][-1]
                deltas.append(current_delta)
                #print(deltas)
            else:
                W_l_1, _ = self.parameters[l+1]
                a_l, z_l = outputs[l]
                deriv = self.derivatives[l](z_l)
                b = W_l_1.T.dot(deltas[-1])
                current_delta = deriv*b
                deltas.append(current_delta)

        deltas = list(reversed(deltas))

        partials = []

        for l in range(self.num_layers):
            _z_l1 = None

            if l==0:
                _z_l1 = X
            else:
                _z_l1 = outputs[l-1][-1]
            dW_l = np.sum(_z_l1[None,:,:].transpose(2,0,1)*deltas[l][None,:,:].transpose(2,1,0),axis=0) \
                + regularizer*self.parameters[l][0]
            db_l = np.sum(deltas[l],axis=1)

            partials.append([dW_l,db_l])

        if gradient_check:
            original_params = list(self.parameters)

            check = 1
            max_grad_diff = 0
            for check in range(self.num_layers):
                W0,b0 = self.parameters[check]
                dd = 10**(-5)
                "check for W"
                numeric_grad = np.zeros(W0.shape)
                for i in range(W0.shape[0]):
                    for j in range(W0.shape[1]):
                        W0_tmp1 = W0.copy()
                        W0_tmp2 = W0.copy()
                        W0_tmp1[i,j] += dd
                        self.parameters[check][0] = W0_tmp1
                        res1 = self.cost(X,Y,regularizer)
                        W0_tmp2[i,j] -= dd
                        self.parameters[check][0] = W0_tmp2
                        res2 = self.cost(X,Y,regularizer)
                        numeric_grad[i,j] = (res1-res2)/(2*dd)

                analytic_grad = partials[check][0]
                res = (analytic_grad-numeric_grad)
                max_grad_diff = max(max_grad_diff,np.abs(res).max())


                "check for b"
                numeric_grad_b = np.zeros(b0.shape)
                analytic_grad_b = partials[check][1]
                for i in range(b0.shape[0]):
                    b0_tmp1 = b0.copy()
                    b0_tmp2 = b0.copy()

                    b0_tmp1[i] += dd
                    self.parameters[check][1] = b0_tmp1
                    res1 = self.cost(X,Y,regularizer)
                    b0_tmp2[i] -= dd
                    self.parameters[check][1] = b0_tmp2
                    res2 = self.cost(X,Y,regularizer)
                    numeric_grad_b[i] = (res1- res2)/(2*dd)
                max_grad_diff = max(max_grad_diff,np.abs(analytic_grad_b-numeric_grad_b).max())
            print("maximum difference between numeric and analytical grad:", max_grad_diff )
            self.parameters = original_params

        return partials

    def fit(self,X,Y, eta=0.0001, momentum =0.5, minibatch=20, max_iter=100, regularizer = 0.2,
            gradient_check = False, cv=None,verbose=False, graphs=False, lbfgs = True):
        if lbfgs:
            self.parameters = self.bfgs(X,Y, regularizer, gradient_check)
        else:
            """u_w, u_b saves momenutm"""
            u_w=[0]*self.num_layers
            u_b=[0]*self.num_layers
            initial_cost = self.cost(X,Y,regularizer)
            print("initial_cost:", initial_cost)

            cost_over_batches_sgd = []
            cost_over_epoch = []
            cost_over_cv_batches = []
            cost_over_cv_epoch = []

            X_cv, Y_cv = (None,None)
            if cv:
                X_cv, Y_cv = cv

            for i in range(max_iter):
                """shuffle data and create batches """
                permutation = np.random.permutation(X.shape[1])
                X = X[:,permutation]
                Y = Y[:,permutation]

                Xs = np.array_split(X,X.shape[1]//minibatch,axis=1)
                Ys = np.array_split(Y,X.shape[1]//minibatch,axis=1)
                batch = 0
                for x,y in zip(Xs,Ys):
                    """dJ/d(parameters) """
                    grad = self.backprop(x,y,regularizer,gradient_check)

                    for l in range(self.num_layers):
                        u_w[l] = momentum*u_w[l] + eta*grad[l][0]
                        self.parameters[l][0] -= u_w[l]
                        u_b[l] = momentum*u_b[l] + eta*grad[l][1]
                        self.parameters[l][1] -= u_b[l]

                    c = self.cost(x,y,regularizer)
                    cost_over_batches_sgd.append(c)
                    if cv:
                        cost_over_cv_batches.append(self.cost(X_cv,Y_cv,regularizer))
                c = self.cost(X, Y, regularizer)
                cost_over_epoch.append(c)
                if cv:
                    cost_over_cv_epoch.append(self.cost(X_cv,Y_cv,regularizer))
                if verbose:
                    print("iteration", i, "COST:", c)
                if graphs:
                    plt.figure()
                    plt.plot(cost_over_batches_sgd,label = 'train')
                    plt.plot(cost_over_cv_batches,label = 'cv')
                    plt.title('cost over batch')
                    plt.legend()
                    plt.figure()
                    plt.plot(cost_over_epoch,label='train')
                    plt.plot(cost_over_cv_epoch,label = 'cv')
                    plt.title('cost of epoch')
                    plt.legend()
                    plt.show()

    def bfgs(self, x, y, regularizer, gradient_check):

        def flatten(wb):
            return np.concatenate((wb[0][0].flatten(),wb[0][1].flatten(),wb[1][0].flatten(),wb[1][1].flatten()))
            # result = np.array(0)
            # for i in range(len(wb)):
            #     for j in range(len(wb[0])):
            #         np.concatenate((result,wb[i][j].flatten()))
            # return result

        def unflatten(theta):
            a,b,c,d = np.split(theta, [self.parameters[0][0].size, self.parameters[0][0].size+self.parameters[0][1].size, self.parameters[0][0].size+self.parameters[0][1].size+self.parameters[1][0].size])
            return [[a.reshape(self.parameters[0][0].shape), b.reshape(self.parameters[0][1].shape)], [c.reshape(self.parameters[1][0].shape), d.reshape(self.parameters[1][1].shape)]]

        def costWrap(wb,X,Y):
            self.parameters = unflatten(wb)
            return self.cost(X,Y,regularizer)

        def backpropWrap(wb,X,Y):
            self.parameters = unflatten(wb)
            return flatten(self.backprop(X,Y, regularizer, gradient_check))

        theta, minf, d = fmin_l_bfgs_b(costWrap,flatten(self.parameters),backpropWrap, [x,y], maxiter=10)
        return unflatten(theta)

def derivative_sigmoid(x):
    """
    :param x assume x = sigmoid(y)
    :return: derivative of sigmoid(y)
    """
    return x*(1-x)

def softmax(X):
    powers = np.exp(X)
    sums = powers.sum(axis=0)
    res = powers/sums[None,:]
    return res

def parta():

    def load(file_name):
        file = np.load(file_name)
        X_train =file['X_train']
        y_train =file['y_train']
        X_test =file['X_test']
        y_test =file['y_test']

        return X_train,y_train,X_test,y_test

    X_train,y_train,X_test,y_test = load('simnim.npz')
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    X_train = X_train.T

    from sklearn.preprocessing import LabelBinarizer
    binarizer = LabelBinarizer()
    binarizer.fit(y_train)
    Y_train_hat = binarizer.transform(y_train)
    Y_train = Y_train_hat.T


    nn = NeuralNetwork([X_train.shape[0],30,Y_train.shape[0]], functions=[sigmoid,softmax], derivatives=[derivative_sigmoid])

    #nn.forward(X)
    #nn.backprop(X,Y,graient_check=True)
    nn.fit(X_train,Y_train,eta=0.01,momentum=0.5,minibatch=32,regularizer=0.2,max_iter=150,gradient_check=False, lbfgs=True)

    output = nn.forward(X_train)

    y_train_output = binarizer.inverse_transform(output.T)
    y_test_output = binarizer.inverse_transform(nn.forward(X_test.T).T)
    print((y_train_output==y_train).mean())
    print((y_test_output ==y_test).mean())


def partb():
    def load(file_name):
        file = np.load(file_name)
        X_train =file['X_train'].T
        y_train =file['y_train']
        X_test =file['X_test'].T
        y_test =file['y_test']
        X_cv =file['X_cv'].T
        y_cv =file['y_cv']

        return X_train,y_train,X_cv,y_cv,X_test,y_test

    train_ = [0,0]
    test_ = [0,0]
    overall = []
    for i in range(14):

        X_train,y_train,X_cv,y_cv,X_test,y_test = load('pofa{}.npz'.format(i))

        from sklearn.preprocessing import LabelBinarizer
        binarizer = LabelBinarizer()
        binarizer.fit(y_train)
        Y_train = binarizer.transform(y_train).T
        Y_cv = binarizer.transform(y_cv).T


#nn.forward(X)
#nn.backprop(X,Y,graient_check=True)

        print(X_train.shape[0], Y_train.shape[0])
        nn = NeuralNetwork([X_train.shape[0],30,Y_train.shape[0]], functions=[sigmoid,softmax], derivatives=[derivative_sigmoid])

        nn.fit(X_train,Y_train,eta=0.01,momentum=0.5,minibatch=16,regularizer=0.15,max_iter=200,gradient_check=False,cv = (X_cv,Y_cv),graphs=False, lbfgs=False)

        output = nn.forward(X_train)

        y_train_output = binarizer.inverse_transform(output.T)
        y_test_output = binarizer.inverse_transform(nn.forward(X_test).T)
        print("Iteration: ",i)
        print((y_train_output==y_train).mean())
        print((y_test_output ==y_test).mean())

        overall.append((y_test == y_test_output).mean())

        train_[0] += (y_train_output==y_train).sum()
        train_[1] += y_train.shape[0]
        test_[0] += (y_test_output==y_test).sum()
        test_[1] += y_test.shape[0]

    print("Average train accuracy: ", train_[0]/train_[1],"Average test accuracy: ",test_[0]/test_[1])
    print(train_,test_)
    overall = np.array(overall)
    print(overall.mean())
partb()






