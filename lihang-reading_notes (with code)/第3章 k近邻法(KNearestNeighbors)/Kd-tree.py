from math import sqrt
from collections import namedtuple
from time import clock
from random import random


class KdNode(object):
    def __init__(self,dom_elt,split,left,right):
        '''
        :param dom_elt: k 维空间的一个样本点
        :param split: 维度分割序号
        :param left: 左子空间
        :param right: 右子空间
        '''
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right


class KdTree(object):
    def __init__(self,data):
        k = len(data[0]) # 数据维度

        def CreateNode(split, data_set): # 按第 split 维划分数据集创建 KdNode
            if  not data_set: # 如果数据为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较

            data_set.sort(key=lambda x:x[split])  # 在 x[split] 这个维度上由小到大的排序
            split_pos = len(data_set) // 2 # 中位数位置
            median = data_set[split_pos]
            split_next = (split +1) % k # 改变维度，

            # 递归的创建 kd 树
            return KdNode(median,split,
                          CreateNode(split_next,data_set[:split_pos]), # 创建左子树
                          CreateNode(split_next,data_set[split_pos+1:])) # 创建右子树

        self.root = CreateNode(0, data)

# Kd 树前序遍历
def preorder(root):
    print(root.dom_elt)
    if  root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)

# 对构建好的 kd 树进行搜索

#
result = namedtuple("Result_tuple","nearest_point nearest_dist nodes_visited")

def find_nearest(tree,point):
    k = len(point) # 数据维度
    def travel (kd_node,target,max_dist):
        if kd_node is None:
            return  result([0]*k,float('inf'),0) # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visitd =1
        s = kd_node.split # 进行维度分割，在构造 kd 树时，每个节点的分割维度已经通过计算定义好了
        pivot = kd_node.dom_elt # 分割的点

        if target[s]  <= pivot[s]:  # 如果在 s 维度目标值小于当前分割点的值
            nearer_node = kd_node.left  # 则目标最可能会出现在左子树中
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right # 反之目标会最可能出现在右子树中
            further_node = kd_node.left

        temp = travel(nearer_node,target,max_dist) # 进行遍历最可能包含目标点的子树区域
        nearest = temp.nearest_point # 此叶节点作为 当前最近点
        dist = temp.nearest_dist # 更新最近距离
        nodes_visitd+=temp.nodes_visited

        if  dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s]) # 第s维上目标点与分割超平面的距离
        if  max_dist < temp_dist:  # 目标点与分割超平面的距离是否大于目前超球体的半径
            return result(nearest,dist,nodes_visitd) # 不相交则可以直接返回，不用继续判断

        # 运行下面的部分说明找到比当前最近邻点还要近的点，更新相关信息
        temp_dist = sqrt(sum((p1 - p2)**2 for p1,p2 in zip(pivot,target)))
        if temp_dist < dist: # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist # 更新最近距离
            max_dist = dist # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node,target,max_dist)
        nodes_visitd+=temp2.nodes_visited # 如果另一个子结点内存在更近距离
        if temp2.nearest_dist < dist:  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest,dist,nodes_visitd)
    return travel(tree.root,point,float('inf')) # 从根节点开始递归

# test

# 例 3.2 前序遍历
data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
kd = KdTree(data)
print(preorder(kd.root))




# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]


# 产生n个k维随机向量
def random_points(k, n):
    return [random_point(k) for _ in range(n)]

ret = find_nearest(kd, [3,4.5])
print (ret)

N = 400000
t0 = clock()
kd2 = KdTree(random_points(3, N))            # 构建包含四十万个3维空间样本点的kd树
ret2 = find_nearest(kd2, [0.1,0.5,0.8])      # 四十万个样本点中寻找离目标最近的点
t1 = clock()
print ("time: ",t1-t0, "s")
print (ret2)