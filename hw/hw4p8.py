import math
import scipy.misc as s


q=5
def m(n):
    if n==1:
        return 2
    else:
        return 2*m(n-1) - s.comb(n-1, q)

N=0
while(2**(N+1) == m(N+1)):
    N+=1;

print N