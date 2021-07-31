# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:41:06 2020

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

short_pos=open("short_reviews/positive.txt", "r").read()
short_neg=open("short_reviews/negative.txt", "r").read()

documents=[]
all_words=[]

# j is adjective, r is adverb, v is verb
#allowed_word_types=["J", "R" , "V"]

allowed_word_types=["J"]
stop_words=set(stopwords.words("english"))

for r in short_pos.split("\n"):
    documents.append( (r, "pos") )
    words=word_tokenize(r)
    pos=nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
    
for r in short_neg.split("\n"):
    documents.append( (r, "neg") )
    words=word_tokenize(r)
    pos=nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
    


'''short_pos_words=word_tokenize(short_pos)
short_neg_words=word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())
    
for w in short_neg_words:
    all_words.append(w.lower())'''
    
save_documents=open("pickeled_algos/Documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()
    
all_words= nltk.FreqDist(all_words)

word_features= list(all_words.keys())[:7000]

save_word_features=open("pickeled_algos/Word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()



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

classifier= nltk.NaiveBayesClassifier.train(training_set)
print("Original NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
        
save_classifier=open("pickeled_algos/OriginalNaiveBayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#MULTINOMIAL
multinomial_nb_classifier= SklearnClassifier(MultinomialNB())
multinomial_nb_classifier.train(training_set)
print("Multinomial NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(multinomial_nb_classifier, testing_set))*100)

save_classifier=open("pickeled_algos/MultinomialNB.pickle", "wb")
pickle.dump(multinomial_nb_classifier, save_classifier)
save_classifier.close()

'''GAUSSIAN
gaussian_nb_classifier= SklearnClassifier(GaussianNB())
gaussian_nb_classifier.train(training_set)
print("Gaussian NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(gaussian_nb_classifier, testing_set))*100)'''

#BERNOULLI
bernoulli_nb_classifier= SklearnClassifier(BernoulliNB())
bernoulli_nb_classifier.train(training_set)
print("Bernoulli NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(bernoulli_nb_classifier, testing_set))*100)

save_classifier=open("pickeled_algos/BernoulliNaiveBayes.pickle", "wb")
pickle.dump(bernoulli_nb_classifier, save_classifier)
save_classifier.close()

#LOGISTIC REGRESSION
logistic_regression_classifier= SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(training_set)
print("logistic_regression_classifier NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(logistic_regression_classifier, testing_set))*100)

save_classifier=open("pickeled_algos/LogisticRegession.pickle", "wb")
pickle.dump(logistic_regression_classifier, save_classifier)
save_classifier.close()

#SGDC
sgdc_classifier= SklearnClassifier(SGDClassifier())
sgdc_classifier.train(training_set)
print("sgdc_classifier NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(sgdc_classifier, testing_set))*100)

save_classifier=open("pickeled_algos/SGDC.pickle", "wb")
pickle.dump(sgdc_classifier, save_classifier)
save_classifier.close()

#SVC
svc_classifier= SklearnClassifier(SVC())
svc_classifier.train(training_set)
print("svc_classifier NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(svc_classifier, testing_set))*100)

save_classifier=open("pickeled_algos/SVC.pickle", "wb")
pickle.dump(svc_classifier, save_classifier)
save_classifier.close()

#LINEAR SVC
linear_svc_classifier= SklearnClassifier(LinearSVC())
linear_svc_classifier.train(training_set)
print("linear_svc_classifier NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(linear_svc_classifier, testing_set))*100)

save_classifier=open("pickeled_algos/LinearSVC.pickle", "wb")
pickle.dump(linear_svc_classifier, save_classifier)
save_classifier.close()

#NU SVC
nu_svc_classifier= SklearnClassifier(NuSVC())
nu_svc_classifier.train(training_set)
print("nu_svc_classifier NaiveBayesClassifier accuracy percentage: ", (nltk.classify.accuracy(nu_svc_classifier, testing_set))*100)

save_classifier=open("pickeled_algos/NuSVC.pickle", "wb")
pickle.dump(nu_svc_classifier, save_classifier)
save_classifier.close()

#FINAL CLASSIFIER
final_classifier= Vote_Classifier(classifier, multinomial_nb_classifier, bernoulli_nb_classifier, logistic_regression_classifier,linear_svc_classifier, sgdc_classifier, nu_svc_classifier)
print("final_classifier  accuracy percentage: ", (nltk.classify.accuracy(final_classifier, testing_set))*100)

#print("Classification : ", final_classifier.classify(testing_set[0][0]), "Confidence percentage: ", final_classifier.confidence(testing_set[0][0]))

def sentiment(text):
    feats=find_features(text)
    return final_classifier.classify(feats)




