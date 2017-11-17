import math

def cost(u: float, v:float) -> float :
    return (u*math.exp(v)-2*v*math.exp(-u))**2

def grad(u: float, v:float) -> (float, float):
    common_part = 2*(u*math.exp(v)-2*v*math.exp(-u))
    return common_part*(math.exp(v)+2*v*math.exp(-u)), common_part*(u*math.exp(v)-2*math.exp(-u))

learning_rate = 0.1
u = 1
v = 1
for i in range(0,30):
    print(cost(u, v))
    u_grad, v_grad = grad(u,v)
    u = u - learning_rate*u_grad
    v = v - learning_rate*v_grad
print(cost(u,v))
print("({}, {})".format(u,v))