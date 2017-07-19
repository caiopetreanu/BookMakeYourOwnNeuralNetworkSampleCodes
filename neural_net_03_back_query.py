#https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part3_neural_network_mnist_backquery.ipynb

import time
print("----- START -----", time.strftime("%c"), "-----")

#http://yann.lecun.com/exdb/mnist/.
#https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network.ipynb

#numpy.random for the normal initialization function normal()
import numpy.random

#scipy.special for the sigmoid function expit()
import scipy.special

import matplotlib.pyplot

%matplotlib inline

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
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

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

    # backquery the neural network
    # we'll use the same termnimology to each item, 
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hideen layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs

input_nodes = 784;
hidden_nodes = 200;
output_nodes = 10;
learning_rate = 0.1;

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

#http://www.pjreddie.com/media/files/mnist_train.csv
#http://www.pjreddie.com/media/files/mnist_test.csv

#load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5;

for e in range(epochs):
    
    # go through all records in the training data set
    for record in training_data_list:

        # split the record by the',' commas
        all_values = record.split(',');
        #image_array = ((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01).reshape((28,28));
        #matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None');

        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01;

        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01;

        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets);
        pass;
    
    pass;

# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# for record in test_data_list:
    # get the test record
    #all_values = record.split(',')
    # print the label
    #print(all_values[0])
    #image_array = ((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01).reshape((28,28));
    #matplotlib.pyplot.figure();
    #matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None');
    #result = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    #print("O teste diz que é", all_values[0])
    #print("A rede neural diz que é", numpy.argmax(result), ", com pontuação de", max(result))
    #pass
    
# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    
    # split the record by the ',' commas
    all_values = record.split(',')
    
    # correct answor is first value
    correct_label = int(all_values[0])
    print(correct_label, "correct_label")
    
    # scale and shifts the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # query the network
    outputs = n.query(inputs)
    
    # the index of the highest value corresponds to the label    
    label = numpy.argmax(outputs)
    print(label, "network's answer")
    
    # append correct or incorrect to the list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

print(scorecard)

#calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance =", scorecard_array.sum() / scorecard_array.size)

print("----- END -----", time.strftime("%c"), "-----")

#https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part3_neural_network_mnist_and_own_data.ipynb

def handwrite_2():
    
    # helper to load data from PNG image files
    import scipy.misc

    # glob helps select multiple files using patterns
    import glob

    # our own image test data set
    #our_own_dataset = []

    # load the png image data as test data set
    for image_file_name in glob.glob('mnist_dataset/mnist_*_?.png'):

        # use the filename to set the correct label
        correct_label = int(image_file_name[-5:-4])

        # load image data from png files into an array
        print ("loading ... ", image_file_name)
        img_array = scipy.misc.imread(image_file_name, flatten=True)

        # reshape from 28x28 to list of 784 values, invert values
        img_data  = 255.0 - img_array.reshape(784)

        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        print(numpy.min(img_data))
        print(numpy.max(img_data))

        # append label and image data to test data set
        #record = numpy.append(label, img_data)
        #our_own_dataset.append(record)

        # query the network
        outputs = n.query(img_data)

        # the index of the highest value corresponds to the label    
        label = numpy.argmax(outputs)

        # append correct or incorrect to the list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            print("correct_label", correct_label, ". Label", label, ". Acertou!")
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            print("correct_label", correct_label, ". Label", label, ". Errou!")
            pass

        matplotlib.pyplot.figure();
        matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')

        pass
    
    pass

def handwrite_1():

    # helper to load data from PNG image files
    import scipy.misc

    # glob helps select multiple files using patterns
    import glob

    # our own image test data set
    #our_own_dataset = []

    # load the png image data as test data set
    for image_file_name in glob.glob('mnist_dataset/mnist_*_?.png'):

        # use the filename to set the correct label
        correct_label = int(image_file_name[-5:-4])

        # load image data from png files into an array
        print ("loading ... ", image_file_name)
        img_array = scipy.misc.imread(image_file_name, flatten=True)

        # reshape from 28x28 to list of 784 values, invert values
        img_data  = 255.0 - img_array.reshape(784)

        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        print(numpy.min(img_data))
        print(numpy.max(img_data))

        # append label and image data to test data set
        #record = numpy.append(label, img_data)
        #our_own_dataset.append(record)

        # query the network
        outputs = n.query(img_data)

        # the index of the highest value corresponds to the label    
        label = numpy.argmax(outputs)

        # append correct or incorrect to the list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            print("correct_label", correct_label, ". Label", label, ". Acertou!")
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            print("correct_label", correct_label, ". Label", label, ". Errou!")
            pass

        matplotlib.pyplot.figure();
        matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')

        pass

    pass

def backwards(l):
    
    # run the network backwards, given a label, see what image it produces

    # label to test
    label = l;
    
    # create the output signals for this label
    targets = numpy.zeros(output_nodes) + 0.01;
    
    # all_values[0] is the target label for this record
    targets[label] = 0.99;
    print(targets);

    # get image data
    image_data = n.backquery(targets);

    # plot image data
    matplotlib.pyplot.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None');

pass