#https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network.ipynb

#numpy.random for the normal initialization function normal()
import numpy.random

#scipy.special for the sigmoid function expit()
import scipy.special

#neural network class definition
class neuralNetwork:
    
    #initialise the neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):

        #set number of nodes in each input, hidden, output layer
        self.inodes = inputNodes;
        self.hnodes = hiddenNodes;
        self.onodes = outputNodes;
        
        #self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        #self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        #link weight matrices, wih and who 
        #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes));
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes));

        #learning rate
        self.lr = learningRate;
        
        #activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass;
    
    #train the neural network
    def train(self, inputs_list, targets_list):
        
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin = 2).T;
        targets = numpy.array(targets_list, ndmin = 2).T;
		
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs);
        
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs);
        
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs);
        
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs);
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs;
        
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors);
        
        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes hidden_errors = numpy.dot(self.who.T, output_errors)
        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs));
        
        #update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs));
        
        pass;
    
    #query the neural network
    def query(self, inputs_list):
        
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin = 2).T
        
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs);
        
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs);
        
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs);
        
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs);

        return final_outputs;
        
    pass;

input_nodes = 3;
hidden_nodes = 3;
output_nodes = 1;
learning_rate = 0.3;

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

# Input: m² (x/200), quartos (x/5), banheiros (x/5)
# Result: R$ (x/1000000), pessoas (x/10), faixa salarial [0,1=normal, 0,2=alto, 0,3=muito alto]

# Treinos

#print("Treino 1");
inputs_list = [0.325, 0.4, 0.2];
train_list = [0.55];
n.train(inputs_list, train_list);
#print("Entrada: ", inputs_list);
#print("Resultado: ", train_list);

#print("Treino 2");
inputs_list = [0.425, 0.6, 0.4];
train_list = [0.75];
n.train(inputs_list, train_list);
#print("Entrada: ", inputs_list);
#print("Resultado: ", train_list);

#print("Treino 3");
inputs_list = [0.325, 0.4, 0.2];
train_list = [0.5];
n.train(inputs_list, train_list);
#print("Entrada: ", inputs_list);
#print("Resultado: ", train_list);

#print("Treino 4");
inputs_list = [0.425, 0.4, 0.4];
train_list = [0.7];
n.train(inputs_list, train_list);
#print("Entrada: ", inputs_list);
#print("Resultado: ", train_list);

#print("Treino 5");
inputs_list = [0.515, 0.8, 0.6];
train_list = [1.05];
n.train(inputs_list, train_list);
#print("Entrada: ", inputs_list);
#print("Resultado: ", train_list);

# Testes

inputs_list = [0.2, 0.4, 0.2];
print("Apartamento de ", inputs_list[0]*200, " m², ", inputs_list[1]*5, " quartos e ", inputs_list[2]*5, " banheiros");
result = n.query(inputs_list);
#print("Entrada: ", inputs_list);
#print("Resultado: ", result);
print("Resultado: R$:", result[0]*1000000);