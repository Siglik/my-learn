import math

# for n in range (380000, 500001, 20000):
#     print n, 4*math.pow(2*n,10)*math.exp(-(n/3200))

N = 5L
r= 0.05
d= 50L
def m(n):
    return 2**n
print 'a', ((8.0/N)*math.log((4*m(2*N))/r))**0.5
print 'b', ((2.0/N)*math.log((2*N*m(N))))**0.5 +  ((2.0/N)*math.log(1/r))**0.5 + 1/N
print 'c', ((1.0/N)*(2 + math.log((6*m(2*N))/r)))**0.5
print 'd', ((0.5/N)*(8 + math.log((4*m(N*N))*20)))**0.5
