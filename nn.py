import numpy as np
import matplotlib.pyplot as plt
import random

class Neuron:
    def __init__(self, inputQuantity=1, bias=1):
        self.inputQuantity = inputQuantity
        self.weights = [1 for _ in range(self.inputQuantity)]
        self.bias = bias

    def sigmoid(self,y):
        return 1/(1+np.exp(-y))

    def output(self, x):
       #print("weights {}".format(self.weights))
       #print("bias {}".format(self.bias))
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
        self.output_matrix=[]

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

    def prepareNetwork(self):
        self.createNetwork()
        for l in self.network:
            l.createLayer()

    def forwardPass(self,x):
        # 2 for input layer and output layer
 
        
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

    def result(self):
        return self.output_matrix[-1]
    
    def mse(self,y_ture, y_pred):
        return np.mean(np.power(y_ture-y_pred , 2))

    def mse_prime(self,y_ture, y_pred):
        return 2 * (y_pred - y_ture) / np.size(y_ture)

    def fit(self,x,y_ture,ep):
        for epoch in range(ep):
            self.forwardPass(x)
            res=self.result()
            print("Output: {}".format(res[0]))
            error1=self.mse(y_ture, res[0])
            print("Error1: {}".format(error1))
            for index_l , l in enumerate(self.network):
                for index_n , n in enumerate(l.layer):
                    # Update each weight in the list
                    n.weights = [random.uniform(6.0, 0) for _ in n.weights]
                    n.bias = random.uniform(6.0, 0)

            
            self.forwardPass(x)
            res=self.result()
            error2=self.mse(y_ture, res[0])
            print("Error2 {}".format(error2))
            if error2 < error1:
                print("Error {}".format(error2))
            if error2 < 0.1:
                break
            

x=[[0,0],[0,1],[1,0],[1,1]]
y=[0,1,1,0]

nn = NeuralNetwork(inputLayerNeuronQuantity=2,\
                    hiddenLayerNeuronQuantity=2,\
                    outputLayerNeuronQuantity=1,\
                    hiddenlayerQuantity=1,\
                    )
nn.prepareNetwork()
for i,data in enumerate(x):
    nn.fit(data,y[i],1)

#print(nn.output_matrix)
    #print(nn.result())
    #print("Error:{}".format(nn.mse(y[i],nn.result()[0])))
#[[np.float64(0.7310585786300049), np.float64(0.7310585786300049)], [np.float64(0.9214430516601156), np.float64(0.9214430516601156)], [np.float64(0.9449497893439537)]]