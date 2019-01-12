import numpy as np
import math

pro_A, pro_B, por_C = 0.5, 0.5, 0.5

data=[1,1,0,1,0,0,1,0,1,1]

def pmf(i, pro_A, pro_B, pro_C):
    pro_1 = pro_A * math.pow(pro_B, data[i]) * math.pow((1-pro_B), 1-data[i])
    pro_2 = pro_A * math.pow(pro_C, data[i]) * math.pow((1-pro_C), 1-data[i])
    return pro_1 / (pro_1 + pro_2)

class EM:
    def __init__(self, prob):
        self.pro_A ,self.pro_B ,self.pro_C = prob

    # E
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow((1 - self.pro_B), 1 - data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow((1 - self.pro_C), 1 - data[i])
        return pro_1*1.0 / (pro_1 + pro_2)

    def fit(self,data):
        count = len(data)
        print('init prob:{},{},{}'.format(self.pro_A,self.pro_B,self.pro_C))
        for d in range(count):
            _ = yield
            tmp_pmf = [self.pmf(k) for k in range(count)]
            pro_A = 1/count * sum(tmp_pmf)
            pro_B = sum([tmp_pmf[k]*data[k] for k in range(count)]) / sum([ tmp_pmf[k] for k in range(count)])
            pro_C = sum([(1-tmp_pmf[k]) * data[k] for k in range(count)]) / sum([(1-tmp_pmf[k]) for k in range(count)])

            print('{}/{} pro_A:{:3f},pro_B:{:3f},pro_C:{:3f}'.format(d+1,count,pro_A,pro_B,pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C

em = EM(prob=[0.5, 0.5, 0.5])
f = em.fit(data)
next(f)

f.send(1)
f.send(2)
