#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:12:30 2020

@author: lalithsrinivas
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

directory = input("Directory: ")
class Decision_tree:
    left_tree = None
    right_tree = None
    attribute = None
    separation_value = None
    left_decision = None
    height = 1
    index=None
    def __init__(self, attribute, separation_value, left_decision):
        self.attribute = attribute
        self.separation_value = separation_value
        self.left_decision = left_decision

    def add_left_tree(self, left_tree):
        try:
            if self.right_tree == None:
                self.height = left_tree.height+1
            else:
                self.height = max(left_tree.height, self.right_tree.height)+1
        except:
            pass
        self.left_tree = left_tree
    
    def add_right_tree(self, right_tree):
        try:
            if self.left_tree == None :
                self.height = right_tree.height+1
            else:
                self.height = max(right_tree.height, self.left_tree.height)+1
        except:
            pass
        self.right_tree = right_tree
    
    def isLeaf(self):
        if self.right_tree == None and self.left_tree == None:
            return True
        else:
            return False
    def predict(self, array, depth=np.inf):
        if self.isLeaf() or depth==0:
            if array[self.attribute] <= self.separation_value:
                return self.left_decision
            else:
                if self.left_decision == 0:
                    return 1
                return 0
        else:
            if array[self.attribute] <= self.separation_value:
                if self.left_tree != None:
                    return self.left_tree.predict(array, depth-1)
                else:
                    return self.left_decision
            else:
                if self.right_tree != None:
                    return self.right_tree.predict(array, depth-1)
                else:
                    if self.left_decision == 0:
                        return 1
                    return 0
    
    def predict_node(self, array, nodes=np.inf):
        if self.isLeaf() or self.index >= nodes:
            # print(self.height)
            if array[self.attribute] <= self.separation_value:
                return self.left_decision
            else:
                if self.left_decision == 0:
                    return 1
                return 0
        else:
            if array[self.attribute] <= self.separation_value:
                if self.left_tree != None:
                    return self.left_tree.predict_node(array, nodes)
                else:
                    # print(self.height)
                    return self.left_decision
            else:
                if self.right_tree != None:
                    return self.right_tree.predict_node(array, nodes)
                else:
                    # print(self.height)
                    if self.left_decision == 0:
                        return 1
                    return 0
                
    def printTree(self, leaf_count, total_count):
        print("attribute: ", self.attribute)
        total_count += 1
        if self.right_tree == None and self.left_tree == None:
            print("leaf", leaf_count+1, "total count", total_count+1)
            return (leaf_count+1, total_count)
        if self.right_tree != None:
            leaf_count, total_count = self.right_tree.printTree(leaf_count, total_count)
        if self.left_tree != None:
            leaf_count, total_count = self.left_tree.printTree(leaf_count, total_count)
        return (leaf_count, total_count)
    
    def printTreeCount(self, leaf_count, total_count):
        # print("attribute: ", self.attribute)
        total_count += 1
        if self.right_tree == None and self.left_tree == None:
            # print("leaf", leaf_count+1, "total count", total_count+1)
            return (leaf_count+1, total_count)
        if self.right_tree != None:
            leaf_count, total_count = self.right_tree.printTreeCount(leaf_count, total_count)
        if self.left_tree != None:
            leaf_count, total_count = self.left_tree.printTreeCount(leaf_count, total_count)
        return (leaf_count, total_count)
    def accuracy(self, array):
        correct = 0
        numEntries = len(array)
        for i in range(numEntries):
            res = self.predict(array[i, :-1], d)
            if res == array[i, -1]:
                correct+=1
        
        return correct/numEntries
    
    def pruneTree(self, array, numNodes= np.inf):
        if not self.isLeaf():
            acc1 = self.accuracy(array)
            temp_left = self.left_tree
            temp_right = self.right_tree
            self.left_tree = None
            self.right_tree = None
            acc2 = self.accuracy(array)
            self.height = 1
            if acc1 - acc2 > -10**-8:
                arr1 = array[array[:, self.attribute] <= self.separation_value]
                arr2 = array[array[:, self.attribute] > self.separation_value]
                if temp_left != None and len(arr1) != 0:
                    self.left_tree = temp_left.pruneTree(arr1)
                    self.height = 1+self.left_tree.height
                if temp_right != None and len(arr2) != 0:
                    self.right_tree = temp_right.pruneTree(arr2)
                    self.height = max(self.height, 1+self.right_tree.height)
                return self
            print("removed a node", self.attribute, acc1, acc2)
        return self
    
    def setIndex(self, count, levelDict, maxHeight):
        if count[maxHeight-self.height] == 0:
            count[maxHeight-self.height] = levelDict[maxHeight-self.height]
            self.index = levelDict[maxHeight-self.height]
        else:
            count[maxHeight-self.height] += 1
            self.index = count[maxHeight-self.height]
        if self.left_tree != None:
            count = self.left_tree.setIndex(count, levelDict, maxHeight)
        if self.right_tree != None:
            count = self.right_tree.setIndex(count, levelDict, maxHeight)
        return count
    def level_count(self, maxIndices, maxHeight):
        # self.index = maxIndices[self.height-1]+1
        maxIndices[maxHeight-self.height] += 1
        if self.right_tree != None:
            maxIndices = self.right_tree.level_count(maxIndices, maxHeight)
        if self.left_tree != None:
            maxIndices = self.left_tree.level_count(maxIndices, maxHeight)
        return maxIndices
def entropy(pres_array):
    result = 0
    try:
        zeros = len(pres_array[pres_array[:, -1] <= 0])
        ones = len(pres_array)-zeros
        if zeros*ones == 0:
                return 0
        p_0 = zeros/(zeros+ones)
        p_1 = 1-p_0
        result = -(p_0*np.log2(p_0)+p_1*np.log2(p_1))
    except:
        return 0
    return result

def mutual_info(B, array, entropy1, medians):
    try: 
        result = 0
        median = medians[B]
        temp1 = array[array[:, B] > median]
        temp2 = array[array[:, B] <= median]
        # if len(temp1) == 0 or len(temp2) == 0:
        #     return None, None, None, None, None
        ent1 = entropy(temp1)
        ent2 = entropy(temp2)
        result = entropy1 - (len(temp1)/len(array))*ent1 - (1-(len(temp1)/len(array)))*ent2
        return result, ent1, ent2, temp1, temp2 
    except:
        return None, None, None, None, None

def build_tree(array, parameters, depth=3, ent=None, last3=[-1, -1]):
    if depth >= 0 and len(array) != 0:
        maxi = -1*np.inf
        attr = -1
        ent1 = 0
        ent2 = 0
        median = np.median(array, axis = 0)
        temp1 = np.array([])
        temp2 = np.array([])
        flag = True
        if ent == None:
            ent = entropy(array)
        for i in last3:
            if i != last3[0]:
                flag = False
        # start = time.time()
        for i in parameters:
            if not (flag and i in last3):
                temp, tent1, tent2, t1, t2 = mutual_info(i, array, ent, median)
                if temp == None:
                    continue
                if(maxi < temp):
                    attr = i
                    ent1, ent2 = tent1, tent2
                    maxi = temp
                    temp1 = t1
                    temp2 = t2
                elif(np.isnan(temp)):
                    print(i, temp)
        # print("time1: ", time.time()-start)
        if attr == -1:
            return None
        last3.pop(0)
        last3.append(attr)
        len1 = len(temp1)
        len2 = len(temp2)
        acc1 = (len(temp2[temp2[:, -1] == 1])+len(temp1[temp1[:, -1] == 0]))/(len1+len2)
        acc2 = 1-acc1
        left = -1
        first = None
        # if depth > 58:
        #     print(attr, depth)
        if (len1 == 0 or len2 == 0) and flag:
            return None
            # print("reached end", attr, depth)
        if acc1 > acc2:
            left = 1
        else:
            left = 0
        try:
            first = Decision_tree(attr, median[attr], left)
            res1 = build_tree(temp2, parameters, depth-1, ent2, last3)
            first.add_left_tree(res1)
            res2 = build_tree(temp1, parameters, depth-1, ent1, last3)
            first.add_right_tree(res2)
            return first
        except Exception as e:
            print("yoo", e)
            if first == None:
                print("tree is none")
            return first
    return None
depth = np.inf
file = open(directory+'/train_x.txt')
y_file = open(directory+'/train_y.txt').read().split('\n')
temp = file.readline().split(' ')
numEntries = (int(temp[0]))
numAttr = int(temp[1][:-1])
array = np.zeros((numEntries, numAttr+1))        #last column contains y values
parameters = {}
for i in range(numEntries):
    array[i][-1] = int(y_file[i])
    temp = file.readline().split(" ")
    for j in range(len(temp)):
        att = temp[j].split(':')
        if int(att[0]) not in parameters:
            parameters[int(att[0])] = 1
        else:
            parameters[int(att[0])] += 1
        array[i][int(att[0])] = float(att[-1])
parameters = sorted(parameters, key=lambda k: parameters[k], reverse=True)        
start_time = time.time()
tree = build_tree(array, parameters, depth)
levels = [0]+tree.level_count([0]*(tree.height), tree.height)
for i in range(1, len(levels)):
    levels[i] += levels[i-1]
tree.setIndex([0]*tree.height, levels, tree.height)
numNodes = tree.printTreeCount(0, 0)[1]
tr_acc = []
nodes = []
for d in range(0, numNodes, 100):
    nodes.append(d)
    correct = 0    
    for i in range(numEntries):
        res = tree.predict_node(array[i, :-1], d)
        if res == array[i, -1]:
            correct+=1
    tr_acc.append(100*correct/numEntries)

te_acc = []
file = open(directory+'/test_x.txt')
y_file = open(directory+'/test_y.txt').read().split('\n')
temp = file.readline().split(' ')
numEntries = (int(temp[0]))
numAttr = int(temp[1][:-1])
array_val = np.zeros((numEntries, numAttr+1))        #last column contains y values
for i in range(numEntries):
    array_val[i][-1] = int(y_file[i])
    temp = file.readline().split(" ")
    for j in range(len(temp)):
        att = temp[j].split(':')
        array_val[i][int(att[0])] = float(att[-1])
for d in range(0, numNodes, 100):
    correct = 0    
    for i in range(numEntries):
        res = tree.predict_node(array_val[i, :-1], d)
        if res == array_val[i, -1]:
            correct+=1
    te_acc.append(100*correct/numEntries)

va_acc = []
file = open(directory+'/valid_x.txt')
y_file = open(directory+'/valid_y.txt').read().split('\n')
temp = file.readline().split(' ')
numEntries = (int(temp[0]))
numAttr = int(temp[1][:-1])
array_val = np.zeros((numEntries, numAttr+1))        #last column contains y values
for i in range(numEntries):
    array_val[i][-1] = int(y_file[i])
    temp = file.readline().split(" ")
    for j in range(len(temp)):
        att = temp[j].split(':')
        array_val[i][int(att[0])] = float(att[-1])
for d in range(0, numNodes, 100):
    correct = 0    
    for i in range(numEntries):
        res = tree.predict_node(array_val[i, :-1], d)
        if res == array_val[i, -1]:
            correct+=1
    va_acc.append(100*correct/numEntries)
plt.plot(nodes, tr_acc, label='Training')
plt.plot(nodes, va_acc, label='Validation')
plt.plot(nodes, te_acc, label='Test')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
##########################################-----------RANDOM FOREST----------##############################################


# old_accuracies = []
file = open(directory+'/test_x.txt')
y_file = open(directory+'/test_y.txt').read().split('\n')
temp = file.readline().split(' ')
numEntries = (int(temp[0]))
numAttr = int(temp[1][:-1])
array_val1 = np.zeros((numEntries, numAttr+1))        #last column contains y values
for i in range(numEntries):
    array_val1[i][-1] = int(y_file[i])
    temp = file.readline().split(" ")
    for j in range(len(temp)):
        att = temp[j].split(':')
        array_val1[i][int(att[0])] = float(att[-1])

clf = RandomForestClassifier()
clf.fit(array[:, :-1], array[:, -1])

params = {'n_estimators':[50, 150, 250, 350, 450], 'max_features':[0.1, 0.3, 0.5, 0.7, 0.9], 'min_samples_split':[2, 4, 6, 8, 10], 'oob_score':[True]}

clf_GS = GridSearchCV(RandomForestClassifier(), params, n_jobs=-1)

clf_GS.fit(array[:, :-1], array[:, -1])
best_params = clf_GS.best_params_
test_acc = []
valid_acc = []
for i in list(params.keys())[:-1]:
        print(i)
        for j in params[i]:
            if i == 'n_estimators':
                temp = RandomForestClassifier(criterion='entropy', n_estimators=j, max_features=0.1, min_samples_split=10, oob_score=True)
                temp.fit(array[:, :-1], array[:, -1])
                test_acc.append(100*temp.score(array_val1[:, :-1], array_val1[:, -1]))
                valid_acc.append(100*temp.score(array_val[:, :-1], array_val[:, -1]))
            elif i == 'max_features':
                temp = RandomForestClassifier(criterion='entropy', n_estimators=250, max_features=j, min_samples_split=10, oob_score=True)
                temp.fit(array[:, :-1], array[:, -1])
                test_acc.append(100*temp.score(array_val1[:, :-1], array_val1[:, -1]))
                valid_acc.append(100*temp.score(array_val[:, :-1], array_val[:, -1]))
            else:
                temp = RandomForestClassifier(criterion='entropy', n_estimators=250, max_features=0.1, min_samples_split=j, oob_score=True)
                temp.fit(array[:, :-1], array[:, -1])
                test_acc.append(100*temp.score(array_val1[:, :-1], array_val1[:, -1]))
                valid_acc.append(100*temp.score(array_val[:, :-1], array_val[:, -1]))

plt.plot(valid_acc[:5], label='n_estimators')
plt.plot(valid_acc[5:10], label="max_features")
plt.plot(valid_acc[10:15], label='min_samples_split')
plt.legend()
plt.xlabel('parameter position')
plt.ylabel('accuracy')
plt.show()

plt.plot(test_acc[:5], label='n_estimators')
plt.plot(test_acc[5:10], label="max_features")
plt.plot(test_acc[10:15], label='min_samples_split')
plt.legend()
plt.xlabel('parameter position')
plt.ylabel('accuracy')
plt.show()

# #############################################---------PRUNING----------####################################################
tree = tree.pruneTree(array_val)
levels = [0]+tree.level_count([0]*(tree.height), tree.height)
for i in range(1, len(levels)):
    levels[i] += levels[i-1]
tree.setIndex([0]*tree.height, levels, tree.height)
numNodes = tree.printTreeCount(0, 0)[1]
tr_acc = []
nodes = []
for d in range(0, numNodes, 100):
    nodes.append(d)
    correct = 0    
    for i in range(len(array)):
        res = tree.predict_node(array[i, :-1], d)
        if res == array[i, -1]:
            correct+=1
    tr_acc.append(100*correct/len(array))

va_acc = []
for d in range(0, numNodes, 100):
    correct = 0    
    for i in range(numEntries):
        res = tree.predict_node(array_val[i, :-1], d)
        if res == array_val[i, -1]:
            correct+=1
    va_acc.append(100*correct/numEntries)

te_acc = []
file = open(directory+'/test_x.txt')
y_file = open(directory+'/test_y.txt').read().split('\n')
temp = file.readline().split(' ')
numEntries = (int(temp[0]))
numAttr = int(temp[1][:-1])
array_val = np.zeros((numEntries, numAttr+1))        #last column contains y values
for i in range(numEntries):
    array_val[i][-1] = int(y_file[i])
    temp = file.readline().split(" ")
    for j in range(len(temp)):
        att = temp[j].split(':')
        array_val[i][int(att[0])] = float(att[-1])
for d in range(0, numNodes, 100):
    correct = 0    
    for i in range(numEntries):
        res = tree.predict_node(array_val[i, :-1], d)
        if res == array_val[i, -1]:
            correct+=1
    te_acc.append(100*correct/numEntries)

plt.plot(nodes, tr_acc, label='Training')
plt.plot(nodes, va_acc, label='Validation')
plt.plot(nodes, te_acc, label='Test')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ##########################################################################################################################
