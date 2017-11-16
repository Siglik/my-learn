import math
import random

def f(x):
    return math.sin(math.pi*x)

a_a= 0.78
sum1=0.0
n=0
for x10000 in range(-10000, 10001):
    n += 1
    x= x10000 / 10000.0
    sum1 += (f(x)- a_a*x)**2

print "bias", (sum1/n)

def find_a(x1, y1, x2, y2):
    return (y1+y2)/(x1+x2)


def var(a1,a2):
    sum = 0.0
    n = 0
    for x1000 in range(-100, 101):
        n += 1
        x = x1000 / 100.0
        sum += (f(x) - a2 * x) ** 2
    return sum/n;


sum1 = 0.0
sum_v = 0.0
for x in range(0, 30000):
    x1= random.random()*2 - 1
    x2 = random.random() * 2 - 1
    a_d = find_a(x1, f(x1), x2, f(x2))
    sum1 += a_d
    sum_v += var(a_d, a_a)

print "avg", sum1/30000.0
print "var", sum_v/30000.0