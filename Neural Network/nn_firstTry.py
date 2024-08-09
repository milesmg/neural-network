import random
import numpy as np
import math
from Data_Cleaning.mnist_importer import wrap_data_file
import os
os.system('cls')


def sigma(x):
    return(1/(1+np.exp(-x)))

def sigprime(x):
    return(sigma(x)*(1-sigma(x)))

data_tuple = wrap_data_file()

class NeuralNet():
    # sizes should be an integer list of #s of neurons in each layer
    def __init__(self, sizes):
        self.sizes = sizes
        self.numLayers = len(sizes)
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.mini_batch_size = None
        self.epochs = None
        self.learning_rate = None

    #split training data (formatted as described in mnist_importer.py) into Randomly assorted mini batches
    #the output is a list of lists of tuples,
    #the larger list has length = num_batches (~50k/mini_batch_size)
    #the internal lists have length = mini_batch size
    #the matrix may have a few extra rows of zeros; might need to be cleaned up before
    def to_mini_batches(self, data):
        batch_size = self.mini_batch_size
        #data = data[0]

        num_batches = math.ceil(len(data) / batch_size)
        #print(f"num_batchs = {num_batches}")
        #print(f"len(data) / self.mini_batch_size) = {len(data) / batch_size}")

        #mini_batches = np.array([[0] * self.mini_batch_size for x in range(num_batches)])
        mini_batches = np.zeros((num_batches, batch_size), dtype='object')

        np.random.shuffle(data)
        i = 0
        j = 0
        for x in range(len(data)):
            mini_batches[i][j] = data[x]
            if j == batch_size - 1:
                j = 0
                i += 1
            else:
                j += 1
        # remove extra 0's -- can't do it bc can't remove elements from ndarrays
        """while i < num_batches:
            while j < batch_size:
                if mini_batches[i][j] == 0:
                    del mini_batches[i][j]
                    print(f"just deleted a 0 at ({i},{j})")
                else:
                    print(f"there wasn't a 0 at ({i},{j})")



        print("minbatchi")
        print()
        print(mini_batches[num_batches-1][self.mini_batch_size-1])
        print()"""
        return(mini_batches)
    
    # takes as input 2-tuple (image, answer)
    # returns tuple: (guessed value = int, cost = float, activations)
    # unnormalized_activations = list of n x 1 ndarrays, where n is 784 (the first layer of neurons) or self.sizes[i]
    def interpret_image(self, image):
        # image should be a tuple (x,y) where x is the vals, y is the vectorized number

        # y is the answer vector
        y = image[1]
        datum = image[0]

        # TEST TO MAKE SURE INDICES WORK
        activations = [0] * (self.numLayers)
        activations[0] = datum
        for x in range(self.numLayers-1):
            datum = np.dot(self.weights[x], datum) + self.biases[x]
            activations[x+1] = datum
            datum = sigma(datum)
        
        #get the guessed value
        maxVal = 0
        guessedVal = None
        for x in range(self.sizes[-1]):
            if datum[x] > maxVal:
                maxVal = datum[x]
                guessedVal = x
        
        """#compute the cost
        diff = y - datum
        sum = np.dot(diff,diff)
        cost = sum / (2)"""
        cost = 1
        
        return(guessedVal, cost, activations)

    #call this fxn to train the model on a single mini batch
    def perform_gradient_descent(self, mini_batch):
        #for each training example:
        #   interpret_data
        #       -- save data for each layer!
        #   compute error in output layer
        #   compute error in each subsequent (previous) layer

        #Note: list_of_activationLists is a list of lists; 
        # each inner list is the list of UNNORMALIZED activations of the various layers of the network from the xth training example; 
        # the outer list iterates through the training examples
            # note that each inner list is itself a list of ndarrays, with each entry in the ndarrays being the (float in [0,1]) UNNORMALIZED activation of a given neuron in that layer
            # so really, it's a list of lists of ndarrays of activations
            # each entry is list_of_activationLists[x][y][z] = the (z+1)th index from the vector describing the unnormalized activation in the (y+1)th layer for the zth training example
        # Similarly, list_of_errorLists is a list of lists; 
        # each entry in the outer list corresponds to a given training example
        # while each inner list corresponds to the error at layer l - n
            # the exact setup should be: list_of_errorLists[x][y][z] = the (z+1)th index from the vector describing the error in the yth layer from the end (the L-y)th layer, from example x
            # we don't need to use the error in the first layer, so y should go from 0 --> self.numLayers - 2 (???)
        
        #create list of activations
        list_of_activationLists = [0]
        printed = False
        for datum_tuple in mini_batch:
            if datum_tuple != 0:
                list_of_activationLists.append(self.interpret_image(datum_tuple)[2])
        
        del list_of_activationLists[0]

        #create list of errors
        list_of_errorLists = [[0] * (self.numLayers - 1) for _ in range(len(list_of_activationLists))]
        which_example = 0
        t = False
        for datum_tuple in mini_batch:
            if datum_tuple != 0:
                #error in last layer; implementing equation 1
                answerDif = sigma(list_of_activationLists[which_example][-1]) - datum_tuple[1]
                """if not t:
                    print(answerDif)
                    print(f"list_of_activationLists[which_example][-1]) = {sigma(list_of_activationLists[which_example][-1])}")
                    print(f"datum_tuple[1] = {datum_tuple[1]}")
                    print(f"")
                    t = not t"""
                list_of_errorLists[which_example][0] = answerDif * sigprime(list_of_activationLists[which_example][-1])
                #error in subsequent layers; implementing equation 2
                for ell in range(1,self.numLayers-1):
                    list_of_errorLists[which_example][ell] = np.dot(np.transpose(self.weights[-ell]), list_of_errorLists[which_example][ell-1]) * sigprime(list_of_activationLists[which_example][-(ell+1)])
                which_example += 1
        num_examples = which_example

        #adjust weights
        #   w_l --> w_l - learning_rate/(num_examples) SUM (over examples) (error(xth example, lth layer) x activation^TRANSPOSE(xth example, l-1th layer)
        for weight in range(len(self.weights)):
            sumForWeights = np.dot(list_of_errorLists[0][-1 - weight], np.transpose(sigma(list_of_activationLists[0][weight])))

            for run in range(1,num_examples):
                sumForWeights += list_of_errorLists[run][-1 - weight]

            self.weights[weight] = self.weights[weight] - (self.learning_rate/num_examples) * sumForWeights

        #adjust biases 
        #   b_l --> b_l - learning_rate/(num_examples) SUM (over examples) error(xth example, lth layer)
        for bias in range(len(self.biases)):
            sumForBiases = list_of_errorLists[0][-1 - bias]
            for run in range(1,num_examples):
                sumForBiases += list_of_errorLists[run][-1 - bias]
            self.biases[bias] = self.biases[bias] - (self.learning_rate/num_examples) * sumForBiases


        



    #call this fxn to train the model on some data
    def train_on_data(self, batched_data):
        #perform_gradient_descent(1)
        for epoch in range(self.epochs):
            print(f"epoch = {epoch}")
            for x in batched_data:
                self.perform_gradient_descent(x)    
    
    #test how well the model works on the validation data
    # input validation data = list of tuples (image,answer)
    # output (# of correct answers, # of images tested)
    def analyze_performance(self, validation_data):
        def vectorToAnswer(vector):
            for x in range(10):
                if vector[x] == 1:
                    return(x)
                
        numCorrect = 0
        for datum in validation_data:
            result = self.interpret_image(datum)
            guessedVal = result[0]
            answer = vectorToAnswer(datum[1])
            if guessedVal == answer:
                numCorrect += 1
        return(numCorrect, len(validation_data))

    #this is the master function; input the data and it will train the model and spit out how well it performed
    def Run(self, data, test_type, epochs, mini_batch_size, learning_rate):
        #set parameters for the given run
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # if test_type == 1, it tests the neural network on a much smaller set of parameters
        # if test_type == 0, it tests the neural network on the full testing_data dataset
        # else it just tests itself raw
        if test_type == 0:
            training_data = self.to_mini_batches(data[0])
            self.train_on_data(training_data)
        elif test_type == 1:
            test_data = self.to_mini_batches(data[2])
            self.train_on_data(test_data) 
        validation_data = data[1]
        analysis = self.analyze_performance(validation_data)
        numCorrect = analysis[0]
        totalTested = analysis[1]
        percentCorrect = numCorrect * 100 / totalTested
        print()
        print(f"Out of {totalTested} cases tested, the model successfully interpreted {numCorrect} for a percent accuracy of {percentCorrect}")
        print()

print()
print("Hello World")
print()

data = wrap_data_file()
test = NeuralNet([784,30,30,30,10])
test.Run(data, 3, 0, 0, 0)
for x in range(100):
    print(f"Now running for 10 epochs. This is pass # {x}.")
    test.Run(data, 0, 10, 5, 10/(x+3))

print()
print("End Test")

