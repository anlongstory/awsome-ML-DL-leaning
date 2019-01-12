import numpy as np

class HiddenMarkov:
    def forward(self,Q,V,A,B,O,PI):
        N = len(Q) # 状态序列大小
        M = len(O) # 观测序列大小
        alphas = np.zeros((N,M))
        T = M # 有几个观测序列就有几个时刻
        for t in range(T): # 遍历每一时刻，算是 alpha 值
            indexOfO = V.index(O[t]) # 找出序列对应的索引
            for i in range(N):
                if t == 0: # 计算初值
                    alphas[i][t] = PI[t][i] * B[i][indexOfO] # P176(10.15)
                    print('alpha1(%d)=p%db%db(o1)=%f' % (i, i, i, alphas[i][t]))
                else:
                    alphas[i][t] = np.dot([alpha[t-1] for alpha in alphas],[a[i] for a in A]) * B[i][indexOfO]
                    print('alpha%d(%d)=[sigma alpha%d(i)ai%d]b%d(o%d)=%f' % (t, i, t - 1, i, i, t, alphas[i][t]))
        P = np.sum([alpha[M-1] for alpha in alphas]) # P176(10.17)

    def backward(self,Q,V,A,B,O,PI): # 后向
        N = len(Q)
        M = len(O)
        betas = np.ones((N,M))
        for i in range(N):
            print('beta%d(%d)=1' % (M, i))

        for t in range(M-2,-1,-1):
            indexOfO = V.index(O[t+1])
            for i in range(N):
                betas[i][t] = np.dot(np.multiply(A[i],[b[indexOfO] for b in B]) , [bata[t+1] for beta in betas])
                realT = t+1
                realI = i+1
                print('beta%d(%d)=[sigma a%djbj(o%d)]beta%d(j)=(' % (realT, realI, realI, realT + 1, realT + 1),
                      end='')
                for j in range(N):
                    print("%.2f*%.2f*%.2f+" % (A[i][j], B[j][indexOfO], betas[j][t + 1]), end='')
                print("0)=%.3f" % betas[i][t])

        indexOfO = V.index(O[0])
        P = np.dot(np.multiply(PI,[b[indexOfO] for b in B]),[beta[0] for beta in betas])
        print("P(O|lambda)=", end="")
        for i in range(N):
            print("%.1f*%.1f*%.5f+" % (PI[0][i], B[i][indexOfO], betas[i][0]), end="")
        print("0=%f" % P)

    def viterbi(self,Q,V,A,B,O,PI):
        N = len(Q)
        M = len(O)
        deltas = np.zeros((N,M))
        psis = np.zeros((N,M))
        I = np.zeros((1,M))
        for t in range(M):
            realT = t+1
            indexOfO = V.index(O[t]) # 找出序列对应的索引
            for i in range(N):
                realI = i+1
                if t == 0:
                    deltas[i][t] = PI[0][i] * B[i][indexOfO]
                    psis[i][t] = 0
                    print('delta1(%d)=pi%d * b%d(o1)=%.2f * %.2f=%.2f'
                          %(realI, realI, realI, PI[0][i], B[i][indexOfO], deltas[i][t]))
                    print('psis1(%d)=0' % (realI))

                else:
                    deltas[i][t] = np.max(np.multiply([delta[t-1] for delta in deltas], [ a[i] for a in A])) * B[i][indexOfO]
                    print('delta%d(%d)=max[delta%d(j)aj%d]b%d(o%d)=%.2f*%.2f=%.5f'
                          % (realT, realI, realT - 1, realI, realI, realT,
                             np.max(np.multiply([delta[t - 1] for delta in deltas], [a[i] for a in A])), B[i][indexOfO],deltas[i][t]))
                    psis[i][t] = np.argmax(np.multiply([delta[t - 1] for delta in deltas], [a[i] for a in A]))
                    print('psis%d(%d)=argmax[delta%d(j)aj%d]=%d' % (realT, realI, realT - 1, realI, psis[i][t]))
        print(deltas)
        print(psis)

        I[0][M-1] = np.argmax([delta[M-1] for delta in deltas])
        print('i%d=argmax[deltaT(i)]=%d' % (M, I[0][M-1]+1))
        for t in range(M-2, -1, -1):
            I[0][t] = psis[int(I[0][t+1])][t+1]
            print('i%d=psis%d(i%d)=%d' % (t+1, t+2, t+2, I[0][t]+1))
        print(I)

# 习题 10.1

Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
# O = ['红', '白', '红', '红', '白', '红', '白', '白']
O = ['红', '白', '红', '白']    #习题10.1的例子
PI = [[0.2, 0.4, 0.4]]

HMM = HiddenMarkov()
# HMM.viterbi(Q,V,A,B,O,PI)

# 习题 10.2

Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '红', '白', '红', '白', '白']
PI = [[0.2, 0.3, 0.5]]

HMM.forward(Q, V, A, B, O, PI)
HMM.backward(Q, V, A, B, O, PI)
