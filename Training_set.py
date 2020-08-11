import pandas as pd
import numpy as np


output_key = ['Input_A6_024', 'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
                      'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
                      'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
                      'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017']

class data_processing:

    def __init__(self, dataset):
        self.dataset = dataset
        print('dataset init successfully ...')

    def output_extract(self):
        output = np.zeros((245, 20))

        for i in range(0, 244):
            for n in range(0, 19):
                output[i][n] = dataset[ output_key[n] ][i]
        return output

    def input_extract(self):

        input = self.dataset
        for i in range(20):
            del input[output_key[i]]

        return input.to_numpy()


class Neural_Network:

    def __init__(self, x, y):
        self.input = x
        np.random.seed(1)
        self.weights = np.random.rand(self.input.shape[1], 20) * 0.0001
        self.labels = y

        print('NN successfully init ...')
        print('Input are\n', self.input, self.input.shape)
        print('weights 1 are\n', self.weights, self.weights.shape)
        print('Labels are ...\n', self.labels)
        print(np.dot(self.input, self.weights))

    def sigmoid(self, x):
        sig = 1 / (1 + np.exp(-x))  # Define sigmoid function
        return sig

    def sigmoid_derivative(self, x):
        return x*(1-x)

    def forward_propagation(self, input, weight):
        return self.sigmoid(np.dot(input, weight))

    def rmse(self, z):
        sum = 0
        for i in range(20):
            for n in range(245):
                sum += (z[n][i] - self.labels[n][i]) ** 2
        loss = np.sqrt((1 / 245) * sum)
        return loss

    def relu(self, Z):
        return np.maximum(0, Z)

    def dRelu(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def training(self):
        lr = 0.05


        for epoch in range(1000):
            # feedforward
            z = self.forward_propagation(self.input, self.weights)
            # backpropagation step 1
            error_out = self.rmse(z)
            error = self.labels - z
            print(error_out.sum())

            # backpropagation step 2
            dcost_dpred = error
            dpred_dz = self.sigmoid_derivative(z)

            z_delta = dcost_dpred * dpred_dz
            self.weights -= lr * np.dot(self.input.T, z_delta)
        return z

if __name__ == '__main__':
    raw_data = pd.read_csv('~/Desktop/Big data/0714train.csv', header = 0, sep=',')

    dataset = pd.read_csv('~/Desktop/Big data/20200810.csv', header = 0)
    processed_data = data_processing(dataset)

    output = processed_data.output_extract()
    input = processed_data.input_extract()


    nn = Neural_Network(input, output)
    z = nn.training()
    print('Prediction is \n---------------------------------------\n', z)
    print('Actual is \n---------------------------------------\n', output)




