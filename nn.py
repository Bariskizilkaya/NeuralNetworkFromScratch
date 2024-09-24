import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size) * 0.1 - 0.05  # Initialize weights in range [-0.05, 0.05]
        self.bias = np.random.rand() * 0.1 - 0.05
        self.output_value = 0
        self.delta = None  # Placeholder for delta during backpropagation

    def output(self, x):
        self.output_value = self.tanh(np.dot(self.weights, x) + self.bias)
        return self.output_value

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - z ** 2

class Layer:
    def __init__(self, neuron_quantity, input_size):
        self.neurons = [Neuron(input_size) for _ in range(neuron_quantity)]

    def forward(self, x):
        return np.array([neuron.output(x) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.layers = []
        self.layers.append(Layer(hidden_layer_size, input_size))
        self.layers.append(Layer(output_size, hidden_layer_size))

    def forward_pass(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, data, target, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            for x, y_true in zip(data, target):
                y_pred = self.forward_pass(x)

                # Calculate the mean squared error (MSE)
                mse = np.mean((y_pred - y_true) ** 2)

                self.backpropagation(x, y_true, learning_rate)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, MSE: {mse}")

    def backpropagation(self, x, y_true, learning_rate):
        # Calculate output layer deltas
        output_layer = self.layers[-1]
        output = np.array([neuron.output_value for neuron in output_layer.neurons])
        
        # Calculate error
        output_error = y_true - output
        for i, neuron in enumerate(output_layer.neurons):
            neuron.delta = output_error[i] * neuron.tanh_derivative(output[i])

        # Backpropagate the error to the hidden layer
        hidden_layer = self.layers[-2]
        hidden_output = np.array([neuron.output_value for neuron in hidden_layer.neurons])

        # Update weights and biases for the output layer
        for i, neuron in enumerate(output_layer.neurons):
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * neuron.delta * hidden_output[j]
            neuron.bias += learning_rate * neuron.delta  # Update bias for output neuron

        # Update weights and biases for the hidden layer
        for i, neuron in enumerate(hidden_layer.neurons):
            neuron.delta = np.dot([n.delta for n in output_layer.neurons], 
                                  [n.weights[i] for n in output_layer.neurons]) * neuron.tanh_derivative(hidden_output[i])
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * neuron.delta * x[j]
            neuron.bias += learning_rate * neuron.delta  # Update bias for hidden neuron

# XOR input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Create and train the neural network
nn = NeuralNetwork(input_size=2, hidden_layer_size=4, output_size=1)  # Increased hidden layer size
nn.fit(X, y, learning_rate=0.1, epochs=10000)

# Test the trained model
for x in X:
    print(f"Input: {x}, Predicted Output: {nn.forward_pass(x)}")
