# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:14:26 2020

@author: MVR
"""

import nltk
import random
#from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class Vote_Classifier(ClassifierI):
        
    def __init__(self, *classifiers):
         self._classifiers = classifiers
        
    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
            
        votes_choice= votes.count(mode(votes))
        confi= float(votes_choice)/ len(votes)
        return confi

documents_f=open("pickeled_algos/Documents.pickle", "rb")
documents=pickle.load(documents_f)
documents_f.close()

word_features_f=open("pickeled_algos/Word_features.pickle", "rb")
word_features=pickle.load(word_features_f)
word_features_f.close()


def find_features(document):
    words=word_tokenize(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
        
    return features

feature_sets=[(find_features(rev), category) for (rev, category) in documents]

random.shuffle(feature_sets)

training_set=feature_sets[:10000]
testing_set=feature_sets[10000:]

#ORIGINAL NAIVE BAYES 
classifier_open=open("pickeled_algos/OriginalNaiveBayes.pickle", "rb")
classifier=pickle.load(classifier_open)
classifier_open.close()

#MULTINOMIAL
classifier_open=open("pickeled_algos/MultinomialNB.pickle", "rb")
multinomial_nb_classifier=pickle.load(classifier_open)
classifier_open.close()

'''GAUSSIAN
gaussian_nb_classifier= SklearnClassifier(GaussianNB())
gaussian_nb_classifier.train(training_set)
print("Gaussian NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(gaussian_nb_classifier, testing_set))*100)'''

#BERNOULLI
classifier_open=open("pickeled_algos/BernoulliNaiveBayes.pickle", "rb")
bernoulli_nb_classifier=pickle.load(classifier_open)
classifier_open.close()

#LOGISTIC REGRESSION
classifier_open=open("pickeled_algos/LogisticRegession.pickle", "rb")
logistic_regression_classifier=pickle.load(classifier_open)
classifier_open.close()

#SGDC
classifier_open=open("pickeled_algos/SGDC.pickle", "rb")
sgdc_classifier=pickle.load(classifier_open)
classifier_open.close()

#SVC
classifier_open=open("pickeled_algos/SVC.pickle", "rb")
svc_classifier=pickle.load(classifier_open)
classifier_open.close()

#LINEAR SVC
classifier_open=open("pickeled_algos/LinearSVC.pickle", "rb")
linear_svc_classifier=pickle.load(classifier_open)
classifier_open.close()

#NU SVC
classifier_open=open("pickeled_algos/NuSVC.pickle", "rb")
nu_svc_classifier=pickle.load(classifier_open)
classifier_open.close()

#FINAL CLASSIFIER
final_classifier= Vote_Classifier(classifier, multinomial_nb_classifier, bernoulli_nb_classifier, logistic_regression_classifier,linear_svc_classifier, sgdc_classifier, nu_svc_classifier)
#print("final_classifier  accuracy percentage: ", (nltk.classify.accuracy(final_classifier, testing_set))*100)

#print("Classification : ", final_classifier.classify(testing_set[0][0]), "Confidence percentage: ", final_classifier.confidence(testing_set[0][0]))

def sentiment(text):
    feats=find_features(text)
    return final_classifier.classify(feats),final_classifier.confidence(feats)




