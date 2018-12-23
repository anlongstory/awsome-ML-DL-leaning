import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# show the original data
# plt.scatter(df[:50]['sepal length'],df[:50]['sepal width'],color='blue',label='0')
# plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'],color='orange', label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()


data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class KNN:
    def __init__(self,X_train,y_train,k=3,p=2):
        '''
        :param k:  k 个最近邻点
        :param p:  距离度量参数
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.p = p

    def predict(self,x):
        # 取出 n 个点
        knn_list = []
        for  i in range(self.k): # 首先取前三个点加入列表中
            dist = np.linalg.norm( x-self.X_train[i],ord=self.p)
            knn_list.append((dist,self.y_train[i]))

        for i in range(self.k , len(self.X_train)): # 然后将余下的点依次与列表中的点距离比较，有更小的则替换
            max_index = knn_list.index(max(knn_list,key=lambda x:x[0])) # 按x[0]即 dist 来找最大值所在索引
            dist = np.linalg.norm(x - self.X_train[i],ord=self.p) # 求范数
            if  knn_list[max_index][0] > dist: # 如果距离更近，就将当前最大值替换
                knn_list[max_index] = (dist,y_train[i])

        # 统计
        knn =[k[-1] for k in knn_list]  # 将最后 K 个点的分类结果新建一个列表
        count_pairs = Counter(knn) # 计算每一个类别分别有几个
        max_count = sorted(count_pairs)[-1] # 将出现次数由小到大排列，取最大的
        return max_count

    def score(self,X_test,y_test):  # 计算准确率
        right_count = 0
        for X,y in zip(X_test,y_test):
            label = self.predict(X)
            if  label == y:
                right_count+=1
        return right_count/len(X_test)


clf = KNN(X_train,y_train)
print(clf.score(X_test,y_test))

test_point = [5.15, 3.0]
print(" The result of classification: ",clf.predict(test_point))

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'],c='blue', label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'],c='orange', label='1')
plt.plot(test_point[0], test_point[1], 'bo',c='red', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()