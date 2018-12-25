### 朴素贝叶斯法

首先朴素贝叶斯法属于**生成模型**，其是基于特征条件独立假设学习输入/输出的联合概率分布，然后基于此模型，对给定输入 x 用贝叶斯定理求出后验概率最大的输出 y。

这里的“朴素”，是因为此方法对条件概率分布作了条件独立性的假设，这是一种很强的假设，所以是 朴素（naive） 。

使用训练数据学习的是 `P（X|Y）` 和 `P（Y）`的估计，最后得到联合分布概率：`P（X,Y）=P(X|Y)*P(Y)`。  

这里假设`P（X|Y）`满足条件独立性：
<img src="https://i.imgur.com/Jx3zdPr.png" width=70%>

于是可以得到：   
<img src="https://i.imgur.com/JSjJcr3.png" width=65%>

最终学习目的是后验概率最大化：

<img src="https://i.imgur.com/4r4EJCM.png" width=70%>

#### 后验概率最大化

0-1损失函数时对期望风险最小化等同于后验概率最大化：


![](https://i.imgur.com/ds6uHBm.png)

#### 参数估计

- 极大似然估计

- 贝叶斯估计  
&emsp;&emsp; 用极大似然估计可能会出现所要估计的概率值为 0 的情况，这会影响到后验概率的计算结果，使分类产生偏差，解决方法就是采用贝叶斯估计。

#### 代码实现  
这部分主要实现模型 高斯朴素贝叶斯：

特征的可能性假设为高斯分布，其概率密度函数为：
![](https://i.imgur.com/IjJ6oLW.png)  
![](https://i.imgur.com/RHixCeu.png)

    # 数学期望
    @staticmethod
    def mean(x):
        return sum(x)/float(len(x))

    # 标准差
    def std(self,x):
        avg = self.mean(x)
        return math.sqrt(sum(math.pow(x_i-avg,2) for x_i in x)/float(len(x)))

    # 概率密度函数
    def gaussian_prob(self,x,mean,std):
        exp = math.pow(math.e, -1*(math.pow(x - mean,2))/(2*std))
        return (1/(math.sqrt(2*math.pi*std)))*exp