import scipy.io as scio
import scipy.stats as scistats
import numpy as np
from operator import itemgetter
import math
import pdb

N_ATTR = 57 #number of attributes

class TreeNode(object):

    def __init__(self, attrIndex, threshold):
        self.attrIndex = attrIndex
        self.threshold = threshold
        self.left = None
        self.right = None

class Leaf(object):

    def __init__(self, classification):
        self.classification = classification

#S is the training matrix
#A is attribute dictionary to be passed recursively
#Note A is negative. Meaning it contains attributes that have been used
def BuildTree(S, A, depth):
    if depth == 0 or isPure(S): #need conditional to check purity
        #pdb.set_trace()
        return Leaf(Classification(S))

    srTup = GetSplits(S, A)
    currentNode = TreeNode(srTup[0], srTup[1])
    newA = list(A)
    newA.append(srTup[0])
    currentNode.left = BuildTree(srTup[2], newA, depth - 1)
    currentNode.right = BuildTree(srTup[3], newA, depth - 1)
    
    return currentNode

def Classification(S):
    numPos = np.count_nonzero(S[:, N_ATTR])
    numNeg = len(S) - numPos
    return 1 if numPos > numNeg else 0

def GetSplits(S, A):
    reductions = []

    for a in range(N_ATTR):
        if a in A:
            continue
        S = S[S[:, a].argsort()]
        thresh = S[0, a] 
        cols = S[:, a]
        for i, t in enumerate(cols):
            if t != thresh:
                thresh = t
                S1 = S[:i]
                S2 = S[i:]
                reductions.append((a, t, S1, S2, UncertaintyReduction(S, S1, S2)))

    return max(reductions, key = itemgetter(4))

def UncertaintyReduction(S, S1, S2):
    uS = Entropy(np.count_nonzero(S[:, N_ATTR]) / len(S)) #num of positives in S
    uS1 = Entropy(0)
    uS2 = Entropy(0)

    if len(S1) != 0:
        uS1 = Entropy(np.count_nonzero(S1[:, N_ATTR]) / len(S1))
    if len(S2) != 0:
        uS2 = Entropy(np.count_nonzero(S2[:, N_ATTR]) / len(S2))

    pS1 = len(S1) / len(S)
    pS2 = len(S2) / len(S)

    return uS - (pS1 * uS1 + pS2 * uS2)

def isPure(S):
    numPos = np.count_nonzero(S[:, N_ATTR])
    return numPos == len(S) or numPos == 0

def Entropy(proportion):
    return scistats.entropy([proportion, 1 - proportion])


def Classify(T, node):
    if type(node) is Leaf:
        return node.classification

    if T[node.attrIndex] < node.threshold:
        return Classify(T, node.left)
    else:
        return Classify(T, node.right)

def ComputeError(data, root):
    error = 0

    for i, row in enumerate(data):
        print(i)
        result = Classify(row, root)
        if result != row[N_ATTR]:
            error += 1

    return error

train_data = scio.loadmat('spam.mat')['train_spam']
test_data = scio.loadmat('spam.mat')['test_spam']

root = BuildTree(train_data, [], 6)
error = ComputeError(test_data, root)
print(error)
