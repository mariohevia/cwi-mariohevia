#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:35:33 2018

@author: mario_hevia
"""

import gensim, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# Used example in:
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

class Word2vec(object):

    def __init__(self, language):
        self.language = language
        if language == 'english':
            # Model from https://code.google.com/archive/p/word2vec/
            self.w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('pretrained_models/GoogleNews-vectors-negative300.bin.gz', 
                                 binary=True)
        else:
            # Model from http://crscardellino.me/SBWCE/
            self.w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('pretrained_models/SBW-vectors-300-min5.bin.gz', 
                                 binary=True)
        nms = 8
        nm = 10
        self.compiled = re.compile('\.|\,|\'|\"|\(|\)|«|»|’')
        self.models = [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025),
                      SVC(gamma=2, C=1), 
#                      GaussianProcessClassifier(1.0 * RBF(1.0)),
                      DecisionTreeClassifier(max_depth=5), 
                      RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                      AdaBoostClassifier(), LogisticRegression(), GaussianNB(), 
                      QuadraticDiscriminantAnalysis(), MLPClassifier(alpha=1)][nms:nm]
        self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", 
#                      "Gaussian Process",
                      "Decision Tree", "Random Forest", "AdaBoost", "Logistic Regression",
                      "Naive Bayes", "QDA", "Neural Net"][nms:nm]
        print('Pretrained w2v loaded \nNumber of models:', len(self.models))
    
    def extract_features(self, word_list):
        for word in word_list: 
            if word not in self.w2vmodel:
#                print(word)
                return True
        word_vecs = [self.w2vmodel.get_vector(word) for word in word_list if word in self.w2vmodel]
        return np.mean(word_vecs, axis=0)

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            target = re.sub(self.compiled, '', sent['target_word'])
            word_list = target.split(' ')
            tmp_feat = self.extract_features(word_list)
            if type(tmp_feat) != type(True):
                X.append(tmp_feat)
                y.append(sent['gold_label'])
        i = 0
        print('sin', k)
        for model in self.models:
            print('Training: ', self.names[i])
            model.fit(X, y)
            i += 1

    def test(self, testset):
        X = []
        i = 0
        flag = 0
        for model in self.models:
            M = []
            for sent in testset:
                target = re.sub(self.compiled, '', sent['target_word'])
                word_list = target.split(' ')
                tmp_feat = self.extract_features(word_list)
                if type(tmp_feat) != type(True):
                    tmp_pred = model.predict(tmp_feat.reshape(1, -1))
                    prediction = tmp_pred[0] == '1'
                else:
                    prediction = True
#                if flag < 2:
#                    flag += 1
#                    print(tmp_feat, "golaaasd\n\n")
                if prediction:
                    M.append('1')
                else:
                    M.append('0')
            X.append((self.names[i], M))
            i += 1
        return X