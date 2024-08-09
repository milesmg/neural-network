import numpy as np
import random
import pickle
import gzip
import os

def open_data_file():
    #Import MNIST data as tuple with (training data, validation data, test data)
    # training data = (images, answers)
    # images = 50,000-entry ndarray, each entry is a 28 x 28 = 784 value ndarray
    # answers = 50,000-entry ndarray, each entry is a number 0-9
    # validation data, test data same except 10k entries 

    #for your troubles:
    relative_data_path = '../Data/mnist.pkl.gz'
    importer_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(importer_directory, relative_data_path)

    f = gzip.open(data_path, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def wrap_data_file():
    #turns each input into a list of n = 50k or 10k 2-tuples, each with (image, answer), where
    #image = a 784 x 1 vector (ndarray)
    #answer = a 10 x 1 vector (ndarray) of 0s and 1 at the correct position


    #turns answer into vector of 0s with 1 at correct position
    def answer_vector(answer):
        answerVector = np.zeros((10,1))
        answerVector[answer] = 1
        return(answerVector)
    
    def wrap_data(data):
        data_images = [np.reshape(x, (784,1)) for x in data[0]]
        data_answers = [answer_vector(x) for x in data[1]]
        return(list(zip(data_images, data_answers)))
    
    data = open_data_file()
    wrapped_data = wrap_data(data[0]), wrap_data(data[1]), wrap_data(data[2])
    return(wrapped_data)