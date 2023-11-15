import numpy as np
from sklearn.model_selection import train_test_split


# Neural Network class
class MyNeuralNetwork:
    activation_functions_ = {
        'relu': {"forward": lambda x: np.maximum(x, 0), "derivative": lambda x: (x > 0) * 1},
        'linear': {"forward": lambda x: x, "derivative": lambda _: 1},
        'sigmoid': {"forward": lambda x: 1 / (1 + np.exp(-x)),
                    "derivative": lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))}
        # TODO: add more activation functions here. We can move the implementations to a separate file.
        #  Also think if this design is good enough, maybe wrap into some classes
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
            # TODO: make it dependent on activation functions. Here, we use He initialization, which is good for ReLu
            self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * np.sqrt(2 / self.n[lay - 1]))

        self.theta = []  # values for thresholds
        # for i = 0 there are no thresholds.
        self.theta.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.theta.append(np.random.randn(layers[lay], 1) * np.sqrt(2 / self.n[lay - 1]))

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

    def fit(self, X, Y):
        # split the data
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=self.percentage_of_validation,
                                                          random_state=42)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        # Due to weights placement, Xs samples have to be columns
        x_train = x_train.T

        self.epoch_error = np.zeros((self.number_of_epochs, 2))
        for epoch in range(0, self.number_of_epochs):
            random_indices = np.random.permutation(x_train.shape[1])
            for sample in random_indices:
                self.feedForward(x_train[:, [sample]])
                self.backpropagation(y_train[sample])

            # self.epoch_error[epoch][0] = self.meanSquaredError(x_train, y_train)
            # self.epoch_error[epoch][1] = self.meanSquaredError(x_val, y_val)

    def meanSquaredError(self, x_data, y_data):
        error_temp = 0
        for i in range(0, len(x_data)):
            self.feedForward(x_data[i])
            for ouput in range(0, self.n[self.L - 1]):
                error_temp += pow(self.xi[self.L - 1][ouput] - y_data[i][ouput], 2)
        return (error_temp * 0.5)

    def feedForward(self, sample):
        # forumla 1 of BP document
        self.xi[0] = sample
        # print(sample.shape)
        for lay in range(1, self.L):
            self.h[lay] = self.w[lay] @ self.xi[lay - 1] - self.theta[lay]
            self.xi[lay] = self.activation(self.h[lay])

    def backpropagation(self, y):
        # calculating deltas for last layer, BP document formula 11
        last_layer_index = self.L - 1
        self.delta[last_layer_index] = (self.activation_derivative(self.h[last_layer_index]) * (self.xi[last_layer_index] - y)).T

        # formula 12
        # we only iterate through layers 1,2,... indextlastyear-1
        for lay in range(last_layer_index - 1, 0, -1):
            der = self.activation_derivative(self.h[lay])
            self.delta[lay] = self.activation_derivative(self.h[lay]).T * (self.delta[lay + 1] @ self.w[lay + 1])

        # formula 14
        for lay in range(1, self.L):
            self.d_w[lay] = -self.learning_rate * (self.xi[lay - 1] @ self.delta[lay]).T + self.momentum * self.d_w_prev[lay]

        for lay in range(1, self.L):
            self.d_theta[lay] = self.learning_rate * self.delta[lay].T + self.momentum * self.d_theta_prev[lay]

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


# layers include input layer + hidden layers + output layer

if __name__ == '__main__':
    turbine = np.genfromtxt("processed_datasets/turbine.csv", dtype=np.float32, delimiter=',', skip_header=1)
    # my_network = MyNeuralNetwork([4, 5, 3, 1], 10, 0.01, 0.9, "relu", 0.2)
    # my_network.fit(turbine[:, :-1], turbine[:, [-1]])

    layers = [4, 5, 1]
    nn = MyNeuralNetwork(layers, 100, 0.01, 0.01, 'relu', 0.1)
    #
    # print("L = ", nn.L, end="\n")
    # print("n = ", nn.n, end="\n")
    #
    # print("xi = ", nn.xi, end="\n")
    # print("xi[0] = ", nn.xi[0], end="\n")
    # print("xi[1] = ", nn.xi[0], end="\n")
    #
    # print("wh = ", nn.w, end="\n")
    # print("wh[1] = ", nn.w[1][0][1], end="\n")
    # print("threshold = ", nn.theta, end="\n")
    #
    nn.fit([[1,2,3,4],[5,6,7,8],[9,10,11,12],[14,15,16,17]],[[1],[2],[3],[4]])



    # nn.fit([[1, 2, 3, 4], [5, 6, 7, 8]], [[1], [2]])
    #
    # errrr = nn.loss_epochs()
    # print("errror", errrr)
    #
    pre = nn.predict(np.array([[10, 10, 10, 10], [5, 6, 7, 8]]))
    print(pre)
    #
    # s = np.array([1, 2, 3, 4])
