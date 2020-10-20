# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:36:54 2019

@author: DELL
"""
__username__ = "aishitdua"

from math import exp
from csv import reader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())



def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
def accuracy_metric(actual, predicted): # for calculating accuracy of each fold
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation i.e. sigmoid
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    """
    Function for forward propogation of the neural network
    """
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output i.e. sigmoid derivative
def transfer_derivative(output):
    return output * (1.0 - output)
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
def draw(y):
    x=[]
    for i in range(100):
        x.append(i)
    plt.plot(x, y, linewidth=2.0)
# Train a network for a fixed number of epochs
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """
    Post training on the dataset, evalutaing the performance on k_folds(n_used here)
    """
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
def train_network(network, train, l_rate, n_epoch, n_outputs):
    """
    training the network using forward propagation and then sending the errors
    into the backpropagation function
    """
    errorsum=[]           # for the graph
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(outputs[i]-expected[i]) for i in range(len(expected))]) # mean square error
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        errorsum.append(sum_error) 
    draw(errorsum )
def predict(network, row): # for test_set
    """
    function used to predict how the model performs on test set
    """
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)
if __name__ == '__main__':
    # Test training backprop algorithm
    seed(50)
    # load and prepare data
    filename = "dataset.csv"
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_int(dataset, i)
    str_column_to_int(dataset, len(dataset[0])-1)
    n_inputs = len(dataset[0]) - 1
    print(n_inputs)
    n_outputs = len(set([row[-1] for row in dataset])) 
    print(n_outputs)
    n_folds=6
    lr=0.3
    n_epoch=100
    n_hidden=11
    # dividing the data into k folds and mean accuracy
    scores = evaluate_algorithm(dataset, back_propagation, n_folds, lr, n_epoch, n_hidden) #

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))