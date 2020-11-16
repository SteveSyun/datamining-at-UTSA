#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:39:38 2020

@author: huaweisun
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




df = pd.read_csv("hwk10.csv", delimiter=",")
print(df)



print("----------------3a-------------------------")
X_train = df[['A','B','C','D','E','F','G']]
y_train = df['H']

test_data = [[-0.85178192,-1.205080166,-0.47429942,0.3131388745,-0.45259607398,-0.3885927570626,-1.408517],
              [-0.546433, 0.233443, 0.4323213, -0.93232, -0.349, 0.90579295, -0.9070617932],
              [-0.4742581598, -0.0954451227, -0.5891579401, -0.2277621998, -0.4227873548, -0.2082422525, -0.8294981286],
              [-0.58889405, -0.82863064, -0.75298597, -0.31603574, -0.76068974, -0.69985870, -0.28024978],
              [0.21921494, -0.52933342, 0.25851981, 0.03601410, 0.33773096, -0.76977946, 0.333333]]

print(test_data)
print("---------------------------------------------\n")
print("---------------3b----------------------------")


def MLPClassifie_def(hidden, activation):
    # Model Kernel
    mlp = MLPClassifier(hidden_layer_sizes = hidden, activation = activation)
    #training
    mlp.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = mlp.predict(test_data)
    return y_pred

print('The predictions unseen data with relu:',MLPClassifie_def(150,'relu'))
print('The predictions unseen data with identity:',MLPClassifie_def(100,'identity'))
print('The predictions unseen data with logistic:',MLPClassifie_def(150,'logistic'))
print('The predictions unseen data with tanh:',MLPClassifie_def(50,'tanh'))
print("------------------------------------------------\n")
print("---------------3c----------------------------")



def SVC_def(a, b):
    # Kernel
    clf = SVC(kernel=a, gamma =b)
    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(test_data)
    return y_pred



print('The predictions unseen data with linear:',SVC_def('linear','scale'))
print('The predictions unseen data with rbf:',SVC_def('rbf','auto'))
print('The predictions unseen data with poly:',SVC_def('poly','auto'))
print('The predictions unseen data with sigmoid:',SVC_def('sigmoid','auto'))

print("--------------------------------------------------\n")
print("---------------3d----------------------------")

def def_KNeighborsClassifier(n_neighbors, algorithm):
    # Linear Kernel
    K_neighbor = KNeighborsClassifier(n_neighbors = n_neighbors,algorithm = algorithm)
    #Train the model using the training sets
    K_neighbor.fit(X_train, y_train)
    
    #Predict the response for test dataset
    y_pred = K_neighbor.predict(test_data)
    return y_pred

print('The predictions unseen data with auto:',def_KNeighborsClassifier(5, 'auto'))
print('The predictions unseen data with ball_tee:',def_KNeighborsClassifier(4,'ball_tree'))
print('The predictions unseen data with kd_tee:',def_KNeighborsClassifier(3,'kd_tree'))
print('The predictions unseen data with brute:',def_KNeighborsClassifier(3,'brute'))



print("report: regarding all the predict data from all different alogrithm\n")
print("the predictions unseen data are almost same either is a list \n")
print("[2 2 2 2 0] or [2 1 1 2 0]")
