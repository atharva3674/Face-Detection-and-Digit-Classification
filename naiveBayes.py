# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naiveBayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    # calculate prior probability of each label
    priorCount = util.Counter()
    self.prior = util.Counter()
    n = len(trainingLabels)
    
    for label in trainingLabels:
      priorCount[label] += 1
      
    for label in self.legalLabels:
      # add one to each legal label's count to ensure there are no zero values - won't mess with Bayes' Rule later
      priorCount[label] += self.k
      self.prior[label] = float(priorCount[label]) / (n + (self.k * len(self.legalLabels)))
  
    # initialize data structure to feature counts (nested dictionaries)
    self.featureCount = util.Counter()
    for f in self.features:
      self.featureCount[f] = util.Counter()
      for label in self.legalLabels:
        self.featureCount[f][label] = {0: self.k, 1: self.k} # intiialize values to k (k represents the initial count) in order to avoid conditional probs becoming 0

    # count how many times each value appears for each feature given a certain label
    for i, datum in enumerate(trainingData):
      for feature in datum:
        self.featureCount[feature][trainingLabels[i]][datum[feature]] += 1  # datum[feature] tells us value of pixel at coordinate 0 or 1

    # turn counts into conditional probabilities
    for f in self.features:
      for label in self.legalLabels:
        total = self.featureCount[f][label][0] + self.featureCount[f][label][1]
        self.featureCount[f][label][0] /= float(total)
        self.featureCount[f][label][1] /= float(total)

        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    # initialize data structures to hold summations of conditional probability for each valid label
    logJoint = util.Counter()
    for label in self.legalLabels:
      logJoint[label] = 0

    # add conditional probabilities up for each valid label
    for label in self.legalLabels:
      logJoint[label] += math.log(self.prior[label])
      for feature in self.features:
        logJoint[label] += math.log(self.featureCount[feature][label][datum[feature]])

    return logJoint
    

    
      
