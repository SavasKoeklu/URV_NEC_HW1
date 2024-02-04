import numpy as np
from sklearn.model_selection import train_test_split


# Neural Network class
class MyNeuralNetwork:
    activation_functions_ = {
        'relu': {"forward": lambda x: np.maximum(x, 0), "derivative": lambda x: (x > 0) * 1},
        'lrelu': {"forward": lambda x: np.where(x > 0, x, x * 0.01), "derivative": lambda x: np.where(x > 0, 1, 0.01)},
        'linear': {"forward": lambda x: x, "derivative": lambda x: np.ones_like(x)},
        'sigmoid': {"forward": lambda x: 1 / (1 + np.exp(-x)),
                    "derivative": lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))},
        'tanh': {"forward": lambda x: np.tanh(x), "derivative": lambda x: 1 - np.tanh(x) ** 2}
    }

    # Todo: chnage name of activation function to "fact"
    def __init__(self, layers, number_of_epochs, learning_rate, momentum, activation_function_name,
                 percentage_of_validation):
        self.activation = self.activation_functions_[activation_function_name]["forward"]
        self.activation_derivative = self.activation_functions_[activation_function_name]["derivative"]
        self.L = len(layers)  # number of Layers
        self.n = layers.copy()  # number of neurons in each layer
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.percentage_of_validation = percentage_of_validation

        self.h = []  # array of array of fields
        # for i = 0 there are no fields.
        self.h.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.h.append(np.zeros(layers[lay]))

        self.xi = []  # node values
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))

        self.w = []  # edge weights
        # for i = 0 there are no weights.
        self.w.append(np.zeros((1, 1)))
        # array is L x numberof neurons in L x number of neurons of L-1
        for lay in range(1, self.L):
            # I eperienced that the proper weight intialisisation is key to that different activations functions can also be used
            if(activation_function_name in ['relu']):   
                # the problem with the relu activation function was the problem of dead Relu, since relu learns only when the field for the last variable is postitive
                # When negative or 0 fileds the relu derivation mapps calculated gradients to 0, so the network can no adjust it's parameter, but this can also while training
                self.w.append(np.random.rand(layers[lay], layers[lay - 1]) * np.sqrt(2 / self.n[lay - 1]))
            
            elif activation_function_name in ['sigmoid','tanh','linear','lrelu']:
                # Initialize weights with Xavier initialization
                # calculate the range for the weights
                lower, upper = -(1.0 / np.sqrt(self.n[lay - 1])), (1.0 / np.sqrt(self.n[lay - 1]))
                self.w.append(np.random.uniform(lower, upper, (layers[lay], layers[lay - 1])))


        self.theta = []  # values for thresholds
        # for i = 0 there are no thresholds.
        self.theta.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.theta.append(np.random.rand(layers[lay],1) * 0.001  )


        self.delta = []  # values for thresholds
        # for i = 0 there are no thresholds.
        self.delta.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.delta.append(np.zeros(layers[lay]))

        self.d_theta = []  # values for changes to the thresholds
        # for i = 0 there are no thresholds, so no changes for them
        self.d_theta.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_theta.append(np.zeros(layers[lay]))

        self.d_theta_prev = []  # values for previous changes to the thresholds
        # for i = 0 there are no thresholds, so no changes for them, so no previous changes
        self.d_theta_prev.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.d_theta_prev.append(np.zeros((layers[lay], 1)))

        self.d_w = []  # change of edge weights
        # for i = 0 there are no weights.
        self.d_w.append(np.zeros((1, 1)))
        # array is L x numberof neurons in L x number of neurons of L-1
        for lay in range(1, self.L):
            self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))

            self.d_w_prev = []  # previois change of edge weights
        # for i = 0 there are no weights, no change of weights so no previios change of weights
        self.d_w_prev.append(np.zeros((1, 1)))
        # array is L x numberof neurons in L x number of neurons of L-1
        for lay in range(1, self.L):
            self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

    def fit(self, X, Y, batch_size=1):
        # split the data
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=self.percentage_of_validation)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        # Due to weights placement, Xs samples have to be columns

        self.epoch_error = np.zeros((self.number_of_epochs, 2))
        for epoch in range(0, self.number_of_epochs):
            random_indices = np.random.permutation(x_train.shape[0])
            #print(f'epoch . {epoch}')
            random_indices = np.random.permutation(len(x_train))
            #print(f'length of random indices is {len(random_indices)}')
            # np.random.shuffle(x_train)
            for i in range(0, len(random_indices), batch_size):
                self.feed_forward(x_train[random_indices[i:i + batch_size], :].T)
                self.backpropagation(y_train[random_indices[i:i + batch_size]].T)

            self.epoch_error[epoch][0] = self.mean_squared_error(x_train, y_train)
            self.epoch_error[epoch][1] = self.mean_squared_error(x_val, y_val)

    def mean_squared_error(self, x_data, y_data):
        prediction = self.predict(x_data)
        return np.mean((prediction.flatten() - y_data.flatten()) ** 2)

    def backpropagation(self, y):
        # calculating deltas for last layer, BP document formula 11

        # experiment with regularization
        # sum_weights = sum([np.sum(np.abs(np.array(layers_weights))) for layers_weights in self.w])
        # Ec = sum_weights * 0.00000001

        last_layer_index = self.L - 1
        self.delta[last_layer_index] = (
                self.activation_derivative(self.h[last_layer_index]) * (self.xi[last_layer_index] - y)).T

        
        # we only iterate through layers 1,2,... indextlastyear-1
        for lay in range(last_layer_index, 0, -1):
            # formula 12
            if lay != last_layer_index:
                #print(f'filed {self.h[lay]}')
                #print(f'{self.activation_derivative(self.h[lay]).T} * ({self.delta[lay + 1]} - {self.w[lay + 1]})')
                self.delta[lay] = self.activation_derivative(self.h[lay]).T * (self.delta[lay + 1] @ self.w[lay + 1])

            # formula 14
            self.d_w[lay] = -self.learning_rate * (self.xi[lay - 1] @ self.delta[lay]).T + self.momentum * \
                            self.d_w_prev[lay]
            self.d_theta[lay] = self.learning_rate * np.sum(self.delta[lay].T, axis=1, keepdims=True) + self.momentum * self.d_theta_prev[lay]
        
        
        #print('activations')
        #print(self.xi)
        #print('these are the calcualted delats')
        #print(self.delta)

        # formula 15, we update all the weights and thresholds:
        for lay in range(1, self.L):
            self.w[lay] += self.d_w[lay]
            self.theta[lay] += self.d_theta[lay]

        # TODO: check if copies are needed here
        self.d_w_prev = [np.copy(arr) for arr in self.d_w]
        self.d_theta_prev = [np.copy(arr) for arr in self.d_theta]

    def loss_epochs(self):
        return self.epoch_error

    def predict(self, X):
        prediction = X.T
        for lay in range(1, self.L):
            prediction = self.activation(self.w[lay] @ prediction - self.theta[lay])
        return prediction.flatten()

    def feed_forward(self, sample):
        # forumla 1 of BP document
        self.xi[0] = sample
        for lay in range(1, self.L):
            self.h[lay] = self.w[lay] @ self.xi[lay - 1] - self.theta[lay]
            self.xi[lay] = self.activation(self.h[lay])



# layers include input layer + hidden layers + output layer

if __name__ == '__main__':
    dataset = np.genfromtxt("processed_datasets/turbine.csv", dtype=np.float32, delimiter=',', skip_header=1)
    #split_index = int(0.8 5* dataset.shape[0])
    

    #network.fit(dataset[:split_index, :-1], dataset[:split_index, -1], batch_size=1)
    #predictions = network.predict(dataset[split_index:, :-1])

    #my_network = MyNeuralNetwork([4, 5, 3, 1], 10, 0.01, 0.9, "relu", 0.2)
    # my_network.fit(turbine[:, :-1], turbine[:, [-1]])

    layers = [4, 8, 32,100, 1]
    nn = MyNeuralNetwork(layers, 30, 0.1, 0.8, 'lrelu', 0.1)
    #
    print("L = ", nn.L, end="\n")
    print("n = ", nn.n, end="\n")
    
    print("xi = ", nn.xi, end="\n")
    print("xi[0] = ", nn.xi[0], end="\n")
    print("xi[1] = ", nn.xi[0], end="\n")
    
    print("wh = ", nn.w, end="\n")
    print("wh[1] = ", nn.w[1][0][1], end="\n")
    print("threshold = ", nn.theta, end="\n")
    #
    nn.fit([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8] ], [[0.5], [0.9] ])

    # nn.fit([[1, 2, 3, 4], [5, 6, 7, 8]], [[1], [2]])
    #
    # errrr = nn.loss_epochs()
    # print("errror", errrr)
    #
    pre = nn.predict(np.array([[0.1, 0.2, 0.3, 0.4],[0.5, 0.6, 0.7, 0.8]]))
    print(pre)
    # s = np.array([1, 2, 3, 4])
