# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:09:15 2018

@author: shrey
"""

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

#Building Model
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})
    
Positive = []
with open(r"C:/Users/shrey/Desktop/pos_tweets.txt") as f:
    for i in f: 
        Positive.append([format_sentence(i), 'Positive'])

Negative = []
with open(r"C:/Users/shrey/Desktop/neg_tweets.txt",encoding="utf8") as f:
    for i in f: 
        Negative.append([format_sentence(i), 'Negative'])
    
training = Positive[:int((0.8)*len(Positive))] + Negative[:int((0.8)*len(Negative))]
test = Positive[int((0.8)*len(Positive)):] + Negative[int((0.8)*len(Negative)):]

#Checking Accuracy
classifier = NaiveBayesClassifier.train(training)
print('***************************************************************************')
print(accuracy(classifier, test))

#Tweets Classification
example1 = "The hard truth about the United States is that the money other countries spend on health and infrastructure, we spend on war."
print(classifier.classify(format_sentence(example1)))

example2 = "Elephant family save drowning calf by pushing it to shallow end of the pool. Wonderful."
print(classifier.classify(format_sentence(example2)))

example3 = "Andrew Gillum has conceded defeat after a recount in the close fought Florida gubernatorial election."
print(classifier.classify(format_sentence(example3)))

example4 = "As of 12pm, AQI remains red. Those with health issues should avoid outdoor activities. AQI may fluctuate."
print(classifier.classify(format_sentence(example4)))

example5 = "Abrams Still Bitter After Election Loss; Refuses To Call Kemp Legitimate Winner"
print(classifier.classify(format_sentence(example5)))

example6 = "#NSFfunded researchers at the @UW uncover an almost 6,000-year record of the West Antarctic Ice Sheet’s motion."
print(classifier.classify(format_sentence(example6)))

example7 = "Incredible to be with our GREAT HEROES today in California. We will always be with you!"
print(classifier.classify(format_sentence(example7)))

example8 = "It was my great honor to host a celebration of Diwali, the Hindu Festival of Lights, in the Roosevelt Room at the @WhiteHouse this afternoon. Very, very special people!"
print(classifier.classify(format_sentence(example8)))

example9 = "Mike Pence's sharp attacks on China are fueling fears of a Cold War that could divide Asia."
print(classifier.classify(format_sentence(example9)))

example10 = "House Democrats are vowing an all-out fight to salvage the Consumer Financial Protection Bureau."
print(classifier.classify(format_sentence(example10)))