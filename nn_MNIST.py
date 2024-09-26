import numpy as np
import time
import pandas as pd


class DNN:
    def __init__(self, sizes=[784, 128, 64, 10], epochs=10, lr=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        input_layer = sizes[0]
        hidden_1 = sizes[1]
        hidden_2 = sizes[2]
        output_layer = sizes[3]

        self.params = {
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            "W2": np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            "W3": np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forward_pass(self, x_train):

        params = self.params
        params["A0"] = x_train

        # Input layer to hidden_1
        params["Z1"] = np.dot(params["W1"], params["A0"])
        params["A1"] = self.sigmoid(params["Z1"])

        # Hidden_1 to hidden_2
        params["Z2"] = np.dot(params["W2"], params["A1"])
        params["A2"] = self.sigmoid(params["Z2"])

        # Hidden_2 to output
        params["Z3"] = np.dot(params["W3"], params["A2"])
        params["A3"] = self.sigmoid(params["Z3"])

        return params["Z3"]

    def backward_pass(self, y_train, output):
        params = self.params

        change_w = {}

        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params["Z3"], derivative=True)
        change_w["W3"] = np.outer(error, params["A2"])

        error = np.dot(params["W3"].T, error) * self.sigmoid(params["Z2"], derivative=True)
        change_w["W2"] = np.outer(error, params["A1"])

        error = np.dot(params["W2"].T, error) * self.sigmoid(params["Z1"], derivative=True)
        change_w["W1"] = np.outer(error, params["A0"])

        return change_w

    def update_weights(self, change_w):

        for key, val in change_w.items():
            self.params[key] -= self.lr * val  # W_t+1 = W_t - lr * ΔW_t

    def compute_accuracy(self, test_data):
        predictions = []
        test_data = test_data.to_numpy()
        for x in test_data:
            inputs = (np.asfarray(x[1:]) / 255.0 * 0.99) + 0.01  # Ensure 784 features
            targets = np.zeros(10) + 0.01
            targets[int(x[0])] = 0.99
            output = self.forward_pass(inputs)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(targets))

        return np.mean(predictions)

    def train(self, train_data, test_data):
        start_time = time.time()

        for epoch in range(self.epochs):
            for i, row in train_data.iterrows():  # Iterate through rows
                data_values = (row.iloc[1:].to_numpy() / 255.0 * 0.99) + 0.01  # Ensure 784 features
                targets = np.zeros(10) + 0.01
                targets[int(row.iloc[0])] = 0.99  # Label for this row
                output = self.forward_pass(data_values)
                change_w = self.backward_pass(targets, output)
                self.update_weights(change_w)

            # Evaluate accuracy on the test data
            accuracy = self.compute_accuracy(test_data)
            print(f"Epoch {epoch + 1}: Accuracy = {accuracy}")

        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")


# Load data
x_train = pd.read_csv('./digit-recognizer/train.csv')
x_test = pd.read_csv('./digit-recognizer/test.csv')

# Create DNN instance
dnn = DNN(sizes=[784, 128, 64, 10], epochs=10, lr=0.001)

# Train the DNN
dnn.train(x_train, x_train)
