import math

def findN(error, M, max_prob):
    return math.log(max_prob/(2*M))/(-2*math.pow(error,2))

print findN(0.05, 1, 0.03)
print findN(0.05, 10, 0.03)
print findN(0.05, 100, 0.03)