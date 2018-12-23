import math
def L_distance(x,y,p):
    sum = 0
    for i in range (len(x)):
        sum += math.pow(abs(x[i]-y[i]),p)
    return math.pow(sum,1/p)

x=[1,1]
y=[4,4]

print(L_distance(x,y,1))
print(L_distance(x,y,2))
print(L_distance(x,y,3))


