from numpy import *

#这里定义T为转移矩阵列代表前一个y(ij)代表由状态i转到状态j的概率,Tx矩阵x对应于时间序列
#这里将书上的转移特征转换为如下以时间轴为区别的三个多维列表，维度为输出的维度
T1=[[0.6,1],[1,0]];T2=[[0,1],[1,0.2]]
#将书上的状态特征同样转换成列表,第一个是为y1的未规划概率，第二个为y2的未规划概率
S0=[1,0.5];S1=[0.8,0.5];S2=[0.8,0.5]
Y=[1,2,2]  #即书上例一需要计算的非规划条件概率的标记序列
Y=array(Y)-1  #这里为了将数与索引相对应即从零开始
P=exp(S0[Y[0]])
for i in range(1,len(Y)):
    P *= exp((eval('S%d' % i)[Y[i]])+eval('T%d' % i)[Y[i-1]][Y[i]])
print(P)
print(exp(3.2))


#这里根据例11.2的启发整合为一个矩阵
F0=S0;F1=T1+array(S1*len(T1)).reshape(shape(T1));F2=T2+array(S2*len(T2)).reshape(shape(T2))
Y=[1,2,2]  #即书上例一需要计算的非规划条件概率的标记序列
Y=array(Y)-1

P=exp(F0[Y[0]])
Sum=P
for i in range(1,len(Y)):
    PIter=exp((eval('F%d' % i)[Y[i-1]][Y[i]]))
    P *= PIter
    Sum += PIter
print('非规范化概率',P)