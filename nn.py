import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, inputQuantity=1, bias=1):
        self.inputQuantity = inputQuantity
        self.weights = [1 for _ in range(self.inputQuantity)]
        self.bias = bias

    def sigmoid(self,y):
        return 1/(1+np.exp(-y))

    def output(self, x):
       y = x * self.weights[0]
       y += self.bias
       return self.sigmoid(y)
    
class Layer:
    def __init__(self, neuronQuantity=1, inputQuantity=1,bias=1):
        self.neuronQuantity = neuronQuantity
        self.inputQuantity = inputQuantity
        self.bias = bias
        self.layer = []

    def createLayer(self):
        self.layer = [Neuron(inputQuantity=self.inputQuantity,bias=self.bias) \
                                           for _ in range(self.neuronQuantity)]

class NeuralNetwork:
    def __init__(self,inputLayerNeuronQuantity=1,\
                      hiddenLayerNeuronQuantity=1,\
                      outputLayerNeuronQuantity=1,\
                      hiddenlayerQuantity=1,\
                      bias=1):
        self.inputLayerNeuronQuantity = inputLayerNeuronQuantity
        self.hiddenLayerNeuronQuantity = hiddenLayerNeuronQuantity
        self.outputLayerNeuronQuantity = outputLayerNeuronQuantity
        self.hiddenlayerQuantity = hiddenlayerQuantity
        self.bias = bias

    def createNetwork(self):
        self.network=[Layer(neuronQuantity=self.hiddenLayerNeuronQuantity,\
                            inputQuantity=self.inputLayerNeuronQuantity,\
                            bias=self.bias) \
                            for _ in range(self.hiddenlayerQuantity)]

        self.network.insert(0,Layer(neuronQuantity=self.inputLayerNeuronQuantity,\
                            inputQuantity=2,\
                            bias=1 ))

        self.network.append(Layer(neuronQuantity=self.outputLayerNeuronQuantity,\
                            # Quantity of 1 step back layer will be the input
                            # quantity of the output layer
                            inputQuantity = self.hiddenLayerNeuronQuantity,bias=1 ))

    def forwardPass(self,x):
        # 2 for input layer and output layer
        self.createNetwork()
        for l in self.network:
            l.createLayer()
        
        # print("layer 0 {}",self.network[0].layer)
        # print("layer 1 {}",self.network[1].layer)
        # print("layer 2 {}",self.network[2].layer)

        self.output_matrix=[]

        for index_layer, l in enumerate(self.network):
            #print(index_layer, l)
            local_matrix=[]
            if index_layer == 0:
                for index_n,n in enumerate(l.layer):
                    local_matrix.append(n.output(x[index_n]))
            else:
                for index_n,n in enumerate(l.layer):
                    preout=np.sum(self.output_matrix[index_layer-1])
                    local_matrix.append(n.output(preout))
            self.output_matrix.append(local_matrix)
            
            #print(self.output_matrix)

    def backpropagate():
        print("asd")


nn = NeuralNetwork(inputLayerNeuronQuantity=2,\
                    hiddenLayerNeuronQuantity=2,\
                    outputLayerNeuronQuantity=1,\
                    hiddenlayerQuantity=1,\
                    )
nn.forwardPass([0,0])

print(nn.output_matrix)