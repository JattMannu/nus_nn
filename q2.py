#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 22:47:54 2018

@author: tanjeremy
"""
def network_forward(network, input_data, label_data=None, phase='train'):
    for layer in network:
        if type(layer) is not SoftmaxOutput_CrossEntropyLossLayer:
            input_data = layer.forward(input_data)
        else:
            layer.eval(input_data, label_data, phase)
    return network

def network_backward(network):
    for layer in reversed(network):
        if type(layer) is SoftmaxOutput_CrossEntropyLossLayer:
            gradient = layer.backward()
        else:
            gradient = layer.backward(gradient)
    return network

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def network_SGD(network, decay=1.0):
    for layer in reversed(network):
        if type(layer) is FullyConnectedLayer:
            layer.lr *= decay
            layer.w -= layer.lr * layer.gW
            layer.b -= layer.lr * layer.gb
        else:
            continue
    return network

def network_momentum_SGD(network, decay=1.0,rho=0.99):
    for layer in reversed(network):
        if type(layer) is FullyConnectedLayer:
            layer.vW = layer.vW * rho + layer.gW
        else:
            continue
    return network

            

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

def sigmoid(x):
    return 1/(1+np.exp(-x))

class FullyConnectedLayer:
    def __init__(self, num_input, num_output, lr=1e-3, scale = 2):
        
        #self.W = np.random.randn(num_output,num_input) * np.sqrt(scale/(num))
        #self.b = np.random.randn(num_output,1) * np.sqrt(scale/(num_input+))
        self.gW = np.zeros(self.W.shape).astype(np.float128)
        self.gb = np.zeros(self.b.shape).astype(np.float128)
        self.gI = np.array([]).astype(np.float128)
        self.vW = np.zeros(self.W.shape).astype(np.float128)
        self.vb = np.zeros(self.b.shape).astype(np.float128)
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
        self.lr = lr
        pass
    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.dot(self.W, input_data) + self.b
        return self.output_data
    def backward(self, gradient_data):
        self.gW = np.dot(gradient_data, np.transpose(self,input_data))
        #self.gb = np.expand_dims(np.mean(gradient_data,axis=1),axis=1)
        self.gI = np.dot(np.transpose(self.W), gradient_data)
        return self.gI
    
class ReLULayer:
    def __init__(self):
        self.gI = np.array([]).astype(np.float128)
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
    def forward(self,x):
        self.input_data = x
        self.output_data = x.clip(0)
        return self.output_data
    def backward(self,gradient):
        self.gI = gradient * (self.input_data > 0).astype(np.float128)
        return self.gI
    
network_1 = [
            FullyConnectedLayer(input_num,100,lr=lr),
            ReLULayer(),
            FullyConnectedLayer(100,40,lr=lr),
            ReLULayer(),
            FullyConnectedLayer(40,output_num,lr=lr),
            SoftmaxOutput_CrossEntropyLossLayer()
        ]

network_2 = [
            FullyConnectedLayer(input_num,28,lr=lr),
            ReLULayer(),
        ]
for p in range(5):
    network_2.append(FullyConnectedLayer(28,28,lr=lr))
    network_2.append(ReLULayer())
network_2.append(FullyConnectedLayer(28,output_num,lr=lr))
network_2.append(SoftmaxOutput_CrossEntropyLossLayer())

network_3 = [
            FullyConnectedLayer(input_num,14,lr=lr,scale=4),
            ReLULayer(),
        ]
for p in range(27):
    network_3.append(FullyConnectedLayer(14,14,lr=lr))
    network_3.append(ReLULayer())
network_3.append(FullyConnectedLayer(14,output_num,lr=lr,scale=4))
network_3.append(SoftmaxOutput_CrossEntropyLossLayer())


