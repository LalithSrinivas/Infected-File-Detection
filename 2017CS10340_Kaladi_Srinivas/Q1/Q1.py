#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:10:16 2020

@author: lalithsrinivas
"""

import pandas as pd
import numpy as np
import time
from sklearn.neural_network import MLPClassifier
 
weightsAtLayers = []
file = pd.read_csv('alphabet/Alphabets/train.csv', header=None).to_numpy()
np.random.shuffle(file)
y_train = np.zeros((len(file), 26))
for i in range(len(file)):
    y_train[i, file[i, -1]] = 1
x_train = np.array(file[:, :-1])

def netJ(layerNum, outputs):
    global weightsAtLayers
    result = 0
    if layerNum != 0:
        result = np.dot(outputs[layerNum-1], weightsAtLayers[layerNum-1])
    return result

def g(x):
    return 1/(1+np.e**(-x))


def delJ(layerNum, delta_down):
    result = np.dot(delta_down, weightsAtLayers[layerNum].T)
    return result

def lastDelJ(outputs, req_out):
    return (req_out-outputs[-1])

weightsAtLayers = []

outputs = []
# print(weightsAtLayers)

def out(x, batch_size):
    global outputs
    outputs[0] = g(np.array(x))
    for j in range(1, len(outputs)):
        if j != len(outputs)-1:
            outputs[j] = np.append(np.ones((batch_size, 1)), g(netJ(j, outputs)), axis=1)
        else:
            outputs[j] = g(netJ(j, outputs))
    return


def sigm_ad(batch_size=100, numAttributes=784, layer_sizes=[100], numClasses=26, learning_rate = 1):
    global weightsAtLayers, outputs
    for i in range(len(layer_sizes)+1):
        if i == 0:
            temp = np.random.normal(0, 0.5, (numAttributes+1, layer_sizes[0]))       # numAttributes rows and layer_sizes[0] columns
            weightsAtLayers.append(temp)
        elif i == len(layer_sizes):
            temp = np.random.normal(0, 0.5, (layer_sizes[i-1]+1, numClasses))
            weightsAtLayers.append(temp)
        else:
            temp = np.random.normal(0, 0.5, (layer_sizes[i-1]+1, layer_sizes[i]))
            weightsAtLayers.append(temp)
    outputs.append(np.zeros((numAttributes+1, )))
    for i in layer_sizes:
        outputs.append(np.zeros((i+1,)))
    outputs.append(np.zeros((numClasses, )))
    for i in range(len(outputs)):
        outputs[i][0] = 1
    outputs = np.array(outputs)
    weightsAtLayers = np.array(weightsAtLayers)
    error = np.inf
    prev_error = np.inf
    count = 0
    while(True):
        # if count%100 == 0:
        #     print(count, prev_error-error, error)
        if (abs(prev_error-error) < (1) and count > 200) or abs(prev_error-error) < (10**-2):
            break
        prev_error = error
        error = 0
        count+=1
        if abs(prev_error-error) < 20:
            learning_rate = 0.5/(count**0.5)
        for s in range(0, len(y_train), batch_size):
            delta_down = []
            # for i in range(batch_size):
            x = np.append(np.ones((batch_size, 1)), x_train[s:s+batch_size, :], axis=1)
            y = y_train[s:s+batch_size, :]
            out(x, batch_size)
            out_diff = outputs*(1-outputs)
            error += sum(sum((y-outputs[-1])**2))
            # print(error)
            delta_down = lastDelJ(outputs, y)*out_diff[-1]
            # outputs += t_outputs
            # BACK PROPAGATION STARTS
            temp_k = (learning_rate/(batch_size))*np.dot(outputs[-2].T, delta_down)
            weightsAtLayers[-1] = np.add(temp_k, weightsAtLayers[-1])
            temp_list = [len(layer_sizes)-1-k for k in range(len(layer_sizes))]
            for j in temp_list:
                temp = delta_down
                delta_down = []
                delta_down = delJ(j+1, temp)*out_diff[j+1]
                delta_down = delta_down[:, 1:]
                temp = (learning_rate/(batch_size))*np.dot(outputs[j].T, delta_down)                        
                # print(outputs[j].shape, delta_down.shape, j, temp.shape, weightsAtLayers[j].shape)
                weightsAtLayers[j] += temp
    # error /= (len(y_train))            
    print("count ", count)
def sigm_nad(batch_size=100, numAttributes=784, layer_sizes=[100], numClasses=26, learning_rate = 1):
    global weightsAtLayers, outputs
    outputs.append(np.zeros((numAttributes+1, )))
    for i in layer_sizes:
        outputs.append(np.zeros((i+1,)))
    outputs.append(np.zeros((numClasses, )))
    for i in range(len(outputs)):
        outputs[i][0] = 1
    outputs = np.array(outputs)
    for i in range(len(layer_sizes)+1):
        if i == 0:
            temp = np.random.normal(0, 0.5, (numAttributes+1, layer_sizes[0]))       # numAttributes rows and layer_sizes[0] columns
            weightsAtLayers.append(temp)
        elif i == len(layer_sizes):
            temp = np.random.normal(0, 0.5, (layer_sizes[i-1]+1, numClasses))
            weightsAtLayers.append(temp)
        else:
            temp = np.random.normal(0, 0.5, (layer_sizes[i-1]+1, layer_sizes[i]))
            weightsAtLayers.append(temp)
    weightsAtLayers = np.array(weightsAtLayers)
    error = np.inf
    prev_error = np.inf
    count = 0
    while(True):
        if count%100 == 0:
            print(count, prev_error-error, error)
        if abs(prev_error-error) < 0.5 :
            break
        prev_error = error
        error = 0
        count+=1
        for s in range(0, len(y_train), batch_size):
            delta_down = []
            # for i in range(batch_size):
            x = np.append(np.ones((batch_size, 1)), x_train[s:s+batch_size, :], axis=1)
            y = y_train[s:s+batch_size, :]
            out(x, batch_size)
            out_diff = outputs*(1-outputs)
            error += sum(sum((y-outputs[-1])**2))
            # print(error)
            delta_down = lastDelJ(outputs, y)*out_diff[-1]
            # outputs += t_outputs
            # BACK PROPAGATION STARTS
            temp_k = (learning_rate/(batch_size))*np.dot(outputs[-2].T, delta_down)
            weightsAtLayers[-1] = np.add(temp_k, weightsAtLayers[-1])
            temp_list = [len(layer_sizes)-1-k for k in range(len(layer_sizes))]
            for j in temp_list:
                temp = delta_down
                delta_down = []
                delta_down = delJ(j+1, temp)*out_diff[j+1]
                delta_down = delta_down[:, 1:]
                temp = (learning_rate/(batch_size))*np.dot(outputs[j].T, delta_down)                        
                # print(outputs[j].shape, delta_down.shape, j, temp.shape, weightsAtLayers[j].shape)
                weightsAtLayers[j] += temp
    # error /= (len(y_train)) 
    print("count ", count)

def g_relu(x):
    # print(x.shape)
    k = x
    for i in range(len(x)):
        k[i] = np.maximum(0, (x[i]+10**-12))
    return k

def out_relu(x, batch_size):
    global outputs
    outputs[0] = g_relu(np.array(x))/(255**2)
    for j in range(1, len(outputs)):
        if j != len(outputs)-1:
            outputs[j] = np.append(np.ones((batch_size, 1)), g_relu(netJ(j, outputs)), axis=1)
        else:
            outputs[j] = g(netJ(j, outputs))
    return

def relu_ad(batch_size=100, numAttributes=784, layer_sizes=[100], numClasses=26, learning_rate = 0.1):
    global weightsAtLayers
    for i in range(len(layer_sizes)+1):
        if i == 0:
            temp = numAttributes+1
            temp = np.random.normal(0, 10, (temp, layer_sizes[0]))       # numAttributes rows and layer_sizes[0] columns
            weightsAtLayers.append(temp)
        elif i == len(layer_sizes):
            temp = layer_sizes[i-1]+1
            temp = np.random.normal(0, 0.05, (temp, numClasses))
            weightsAtLayers.append(temp)
        else:
            temp = layer_sizes[i-1]+1
            temp = np.random.normal(0, 0.05, (temp, layer_sizes[i]))
            weightsAtLayers.append(temp)
    weightsAtLayers = np.array(weightsAtLayers)
    global outputs
    outputs.append(np.zeros((numAttributes+1, )))
    for i in layer_sizes:
        outputs.append(np.zeros((i+1,)))
    outputs.append(np.zeros((numClasses, )))
    for i in range(len(outputs)):
        outputs[i][0] = 1
    outputs = np.array(outputs)
    error = np.inf
    prev_error = np.inf
    count = 0
    while(True):
        # if count%1 == 0:
        #     print(count, prev_error-error, error)
        if (abs(prev_error-error) < 0.5 and count > 200) or abs(prev_error-error) < 10**-10 :
            break
        prev_error = error
        error = 0
        count+=1
        # learning_rate = (0.5/(int(count)**0.5))
        # print(learning_rate, count)
        for s in range(0, len(y_train), batch_size):
            delta_down = []
            # for i in range(batch_size):
            x = np.append((np.ones((batch_size, 1))), x_train[s:s+batch_size, :], axis=1)
            y = y_train[s:s+batch_size, :]
            out_relu(x, batch_size)
            out_diff = outputs*(1- outputs)
            for j in range(len(outputs)-1):
                out_diff[j] = 0.5*(np.sign((outputs[j]-10**-12))+1)
            out_diff[-1] = outputs[-1]*(1-outputs[-1])
            outputs = g_relu(outputs)
            error += sum(sum((y-outputs[-1])**2))
            # print(error)
            delta_down = lastDelJ(outputs, y)*out_diff[-1]
            # outputs += t_outputs
            # BACK PROPAGATION STARTS
            temp_k = (learning_rate/batch_size)*np.dot(outputs[-2].T, delta_down)
            weightsAtLayers[-1] = np.add(temp_k, weightsAtLayers[-1])
            temp_list = [len(layer_sizes)-1-k for k in range(len(layer_sizes))]
            for j in temp_list:
                temp = delta_down
                delta_down = []
                delta_down = delJ(j+1, temp)*out_diff[j+1]
                delta_down = delta_down[:, 1:]
                # print(outputs[j].shape, delta_down.shape, j, temp.shape, weightsAtLayers[j].shape)
                weightsAtLayers[j] += (learning_rate/batch_size)*np.dot(outputs[j].T, delta_down)
    # error /= (len(y_train)) 
    print("count ", count)
layers = [1, 5, 10, 50, 100]
file1 = pd.read_csv('alphabet/Alphabets/test.csv', header=None).to_numpy()
np.random.shuffle(file1)
y_test = np.zeros((len(file1), 26))
for i in range(len(file1)):
    y_test[i, file1[i, -1]] = 1
x_test = np.array(file1[:, :-1])
print("Addaptive")
for l in layers:       
    weightsAtLayers = []       
    outputs = []      
    start = time.time()
    sigm_ad(layer_sizes=[l])
    print("time taken: ", time.time()-start)
    
    out(np.append(np.ones((len(x_train), 1)), x_train, axis=1), len(x_train))
    
    predicts = np.argmax(outputs[-1], axis=1)
    
    acc = 100*(np.count_nonzero(predicts-np.argmax(y_train, axis=1))/13000)
    
    print("training accuracy with ", l, "units at hidden layer:", 100-acc)
    outputs = outputs*0
    out(np.append(np.ones((len(x_test), 1)), x_test, axis=1), len(x_test))
    predicts = np.argmax(outputs[-1], axis=1)
    
    acc = 100*(np.count_nonzero(predicts-np.argmax(y_test, axis=1))/6500)
    
    print("test accuracy with ", l, "units at hidden layer:", 100-acc)
    
print("ReLU")
       
weightsAtLayers = []       
outputs = []      
start = time.time()
relu_ad(layer_sizes=[100, 100])
print("time taken: ", time.time()-start)

out_relu(np.append(np.ones((len(x_train), 1)), x_train, axis=1), len(x_train))

predicts = np.argmax(outputs[-1], axis=1)

acc = 100*(np.count_nonzero(predicts-np.argmax(y_train, axis=1))/13000)

print("training accuracy with ", [100, 100], "units at hidden layer:", 100-acc)
out_relu(np.append(np.ones((len(x_test), 1)), x_test, axis=1), len(x_test))
predicts = np.argmax(outputs[-1], axis=1)

acc = 100*(np.count_nonzero(predicts-np.argmax(y_test, axis=1))/6500)

print("test accuracy with ", [100, 100], "units at hidden layer:", 100-acc)
    
print("sigmoid")
       
weightsAtLayers = []       
outputs = []      
start = time.time()
sigm_ad(layer_sizes=[100, 100])
print("time taken: ", time.time()-start)

out(np.append(np.ones((len(x_train), 1)), x_train, axis=1), len(x_train))

predicts = np.argmax(outputs[-1], axis=1)

acc = 100*(np.count_nonzero(predicts-np.argmax(y_train, axis=1))/13000)

print("training accuracy with ", [100, 100], "units at hidden layer:", 100-acc)
out(np.append(np.ones((len(x_test), 1)), x_test, axis=1), len(x_test))
predicts = np.argmax(outputs[-1], axis=1)

acc = 100*(np.count_nonzero(predicts-np.argmax(y_test, axis=1))/6500)

print("test accuracy with ", [100, 100], "units at hidden layer:", 100-acc)
print("Non Addaptive")
for l in layers:       
    weightsAtLayers = []       
    outputs = []      
    start = time.time()
    sigm_nad(layer_sizes=[l])
    print("time taken: ", time.time()-start)
    
    out(np.append(np.ones((len(x_train), 1)), x_train, axis=1), len(x_train))
    
    predicts = np.argmax(outputs[-1], axis=1)
    
    acc = 100*(np.count_nonzero(predicts-np.argmax(y_train, axis=1))/13000)
    
    print("training accuracy with ", l, "units at hidden layer:", 100-acc)
    outputs = outputs*0
    outputs = np.array(outputs)
    out(np.append(np.ones((len(x_test), 1)), x_test, axis=1), len(x_test))
    predicts = np.argmax(outputs[-1], axis=1)
    
    acc = 100*(np.count_nonzero(predicts-np.argmax(y_test, axis=1))/6500)
    
    print("test accuracy with ", l, "units at hidden layer:", 100-acc)
    
clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', learning_rate='invscaling', batch_size=100, learning_rate_init=0.1, max_iter=2000)

clf.fit(x_train/255, y_train)
print("MLP Classifier test accuracy", clf.score(x_test/255, y_test)*100)
print("MLP Classifier test accuracy", clf.score(x_train/255, y_train)*100)

