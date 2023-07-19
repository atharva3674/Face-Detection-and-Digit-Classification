import classificationMethod
import numpy as np
import heapq
from numpy import dot
from numpy.linalg import norm

class KNearestClassifier(classificationMethod.ClassificationMethod):

    def __init__(self, legalLabels, neighbors):
        self.legalLabels = legalLabels
        self.type = "kNearest"
        self.k = neighbors # this is the number of neighbors we are looking at to compare the test data to find the label

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = trainingData[0].keys()
        self.features.sort()

        # read in each datum as a coordinate in vector-form
        trainPoints = []

        for index, datum in enumerate(trainingData):
            trainPoint = []

            # build the vector for each datum
            for feature in self.features:
                trainPoint.append(datum[feature]) 

            trainPoints.append((trainPoint, trainingLabels[index]))

        self.trainPoints = trainPoints


    def classify(self, testData):
        # read in each datum as a coordinate in vector-form
        testPoints = []

        for index, datum in enumerate(testData):
            testPoint = []
            
            for feature in self.features:
                testPoint.append(datum[feature]) 

            testPoints.append(testPoint)
            
        self.testPoints = testPoints
        
        # classification occurs here
        choices = []

        for testPoint in testPoints:
            neighbors = self.getNeighbors(testPoint, self.k)
            choices.append(self.mostVoted(neighbors))

        return choices
    

    # gets k closest neighbors to a given point
    def getNeighbors(self, testPoint, k):
        distances = []

        for trainPoint, trainLabel in self.trainPoints:
            distances.append((self.calculateCosineDistance(trainPoint, testPoint), trainLabel))
        
        #distances.sort(key=lambda x: x[0])
        heapq.heapify(distances)

        res = []
        for _ in range(k):
            res.append(heapq.heappop(distances)[1])

        return res
    
    # returns element that occurs most frequently in list
    def mostVoted(self, nums):
        counts = {}
        for num in nums:
            if num not in counts:
                counts[num] = 1
            else:
                counts[num] += 1
        return max(counts, key=counts.get)

    # this function should receive two lists that represent coordinates in vector-form
    def calculateEuclidean(self, p, q):
        sumOfSquares = 0
        for p_, q_ in zip(p, q):
            sumOfSquares += ((q_ - p_)**2)
        return np.sqrt(sumOfSquares)
    
    # this function should receive two lists that represent coordinates in vector-form
    def calculateManhattan(self, p, q):
        sumOfDiffs = 0
        for p_, q_ in zip(p, q):
            sumOfDiffs += abs(q_ - p_)
        return sumOfDiffs
    
    # this function should receive two lists that represent coordinates in vector-form
    def calculateCosineSimilarity(self, p, q):
        return (dot(p, q)/(norm(p)*norm(q)))
    def calculateCosineDistance(self, p, q):
        return (1 - self.calculateCosineSimilarity(p, q))