#coding=utf-8
import pandas as pd
import numpy as np

'''
2019-11-22
by gmk
    1 读数据
    2 把特征按取值划分，输出这样划分的分类情况列
    3 统计该特征该数值的分类情况
    4 算gini和gini增益
    5 生成树及迭代
    6 尝试预测
    7 进行剪枝   ①算 Rt ②算 RTt ③算 α ④找该树最小α ⑤循环剪去最小α生成一个子树列表  ⑥根据测试集选择出最佳子树
    #下面是可以进行的优化
    Ⅰ   连续型特征变为离散型：在划分的时候，首先写一个函数把连续型特征变成离散型分成两类，用gini判别
    Ⅱ   变成离散型之后，就可以把整个数据集标准化了，即变成0 1 2 3等等
    Ⅲ   如果分类情况超过两种，计算一下   各种合并成2种分类  的 gini系数，  取最小的那种
    Ⅳ   然后如果某特征取值超过两种，就先把这个特征 （非合并） 分为两种取值 然后得到的各种column 分别计算完gini值 
    再比较得到最小的那种，最后输出
    Ⅴ   可以生成一个树图     
    
    

'''
class BinNode:
    #结点类
    def __init__(self, data, left=None, right=None, position = None):
        self.data = data
        self.left = left
        self.right = right
        self.position = position
        self.depth = None
        
    def setLeft(self, left):
        self.left = left
    def setRight(self, right):
        self.right = right
    def setPosition(self, position):
        self.position = position
    def preTraverse(self):
        #前序遍历从该结点开始的树
        if len(self.data):
            print('该结点内容为：\n',self.data)
            if self.position != None and self.left:
                print('划分依据为第%d个特征（显示时加了1）'%(self.position+1))
        else:
            return
        if self.left:
            print('左边')
            self.left.preTraverse()
        if self.right:
            print('右边')
            self.right.preTraverse()
    def leavesNum(self):
        #统计子结点数
        num = 0
        if self.left:
            num = num + self.left.leavesNum()
        else:
            num = num + 1
            return num
        if self.right:
            num = num + self.right.leavesNum()
        else:
            return
        return num
    def rootDepth(self):
        depth = 0
        if self.left:
            depth = depth + 1
            depth_left = self.left.rootDepth()
            depth_right = self.right.rootDepth()
            depth = depth + max(depth_left,depth_right)
            return depth
        else:
            return depth
    def predict(self):
        counts = np.bincount(self.data[:,-1])
        predict_class = np.argmax(counts)
        return predict_class

def readData(dataset_name):
    '''
    函数功能：读取数据集里面的数据
    输入参数：数据集名称
    输出参数：数据集
    数据集的最后一列为分类情况
    '''
    raw_dataset = pd.read_csv("%s"%dataset_name, sep=',', header=0, encoding = 'gb2312') 
    raw_dataset = np.array(raw_dataset)
    return raw_dataset



def divideFeature(feature_column,class_column):
    '''
    函数功能：将该特征 根据取值的不同，划分为left right两类
    输入参数：特征列
    输出参数：特征列根据取值不同划分后，所对应的分类列
    '''
    left = []
    right = []
    value_class = np.unique(feature_column)
    for i in range(len(feature_column)):
        if feature_column[i] == value_class[0]:
            left.append(class_column[i])
        elif feature_column[i] == value_class[1]:
            right.append(class_column[i])
    print('左边对应的分类情况为：',left)
    print('右边对应的分类情况为：',right) 
    return left,right  #这边left和right应该是输出划分的该特征的这种数值下，对应的分类情况


def countValue(column):
    '''
    函数功能：统计 任何 分类情况列，两种分类出现的次数
    输入参数：上一个函数输出的分类情况，或者是原数据集的分类情况
    输出参数：L：列长度； num：统计两种分类出现次数的列表
    '''
    class_class = np.unique(column) #目前是2种，＞2种也要变成2类，怎么变之后再处理
    num = np.zeros(2)  
    L = len(column)
    for i in range(L):
        if column[i] == class_class[0]:
            num[0] = num[0] + 1
        elif column[i] == class_class[1]:
            num[1] = num[1] + 1
    print('这种取值分为第一类的有%d个'%num[0])
    print('这种取值分为第二类的有%d个'%num[1])
    return L, num

def getGini(column):
    '''
    函数功能：根据分类列，计算Gini系数
    输入参数：分类列
    输出参数：分类的Gini系数
    '''
    L,num = countValue(column)
    if L == 0:
        gini = 0
    else:
        gini = 1 - (pow(num[0]/L,2) + pow(num[1]/L,2))
    print('gini:',gini)
    return gini

def GiniGain(class_column,left,right):
    '''
    函数功能：计算Gini增益
    输入参数：原数据集的分类列，如果以这个特征进行分类的话，左右两种取值对应的分类列
    输出参数：Gini系数增益，用来判断以哪个特征划分数据集
    '''
    L = len(class_column)
    L_left = len(left)
    L_right = len(right)
    gini_sum = (L_left/L)*getGini(left) + (L_right/L)*getGini(right)
    gain = getGini(class_column) - gini_sum
    print('该特征的Gini系数增益为：',gain)
    return gain

def divideDataSet(dataset,position):
    '''
    函数功能：已经确定了以哪个特征划分数据集之后，进行数据集的划分
    输入参数：原数据集，position：划分特征的位置
    输出参数：划分后的左右枝
    '''
    divide_column = dataset[:,position]
    value_class = np.unique(divide_column)
    left_dataset = []
    right_dataset = []
    
    for i in range(dataset.shape[0]):
        if divide_column[i] == value_class[0]:
            left_dataset.append(dataset[i])
        elif divide_column[i] == value_class[1]:
            right_dataset.append(dataset[i])

    left_dataset = np.array(left_dataset)
    right_dataset = np.array(right_dataset)
    print('划分后左边的数据集为：\n',left_dataset)
    print('划分后右边的数据集为：\n',right_dataset)
    return left_dataset,right_dataset

def generateTree(dataset):
    '''
    这是主要的生成树的函数
    函数功能：调用上述的函数，将原始数据集生成决策树
    输入参数：原数据集
    输出参数：决策树
    '''
    class_column = np.array(dataset[:,-1])
    gain = np.zeros(dataset.shape[1]-1)
    for i in range(dataset.shape[1]-1):
        print('下面是第%d个特征(显示时加了1)'%(i+1))
        left,right = divideFeature(dataset[:,i],dataset[:,-1])
        gain[i] = GiniGain(class_column, left, right)
    max_position = gain.argmax()
    print('gain增益最大为：',gain.max())
    print('gain增益最大的特征是第%d个(显示时加了1)'%(max_position+1))
    left_dataset,right_dataset = divideDataSet(dataset, max_position) #即使gain≯0 也先把左右枝杈分出来吧
    dataset = BinNode(dataset)
    if gain.max() > 0 :  #and len(dataset) >= 2
        dataset.setPosition(max_position)
        print('下面对左边继续分枝')
        left_dataset = generateTree(left_dataset)
        print('下面对右边继续分枝')
        right_dataset = generateTree(right_dataset)

        dataset.setLeft(left_dataset)
        dataset.setRight(right_dataset)
        return dataset
    else: 
        print('这一枝划分完成，这一次计算结果不分枝')
        return dataset


def predictClass(features,tree_node):
    '''
    函数功能：预测分类
    输入参数：features：需要预测的数据的特征; tree_node:树的根结点
    输出参数：预测的分类结果
    '''
    if tree_node.position != None and tree_node.left:
        if features[tree_node.position] == 0:
            predict_class = predictClass(features,tree_node.left)
            return predict_class
        elif features[tree_node.position] == 1:
            predict_class = predictClass(features,tree_node.right)
            return predict_class
    else:
        #以分类的众数代表预测分类
        counts = np.bincount(tree_node.data[:,-1])
        predict_class = np.argmax(counts)
        return predict_class

##################################
#下面开始使用代价复杂度（CCP）剪枝
def predictCorrectRate(test_set,tree):
    '''
    函数功能：计算预测的正确率
    输入参数：test_set:测试集； tree：决策树
    输出参数：预测的正确率
    '''
    correct_num = 0
    for i in range(test_set.shape[0]):
        if predictClass(test_set[i,0:-1], tree) == test_set[i,-1]:
            correct_num = correct_num + 1
    correct_rate = correct_num/test_set.shape[0]
    return correct_rate


def predictErrorRt(tree_node,sample_size):
    '''
    函数功能：计算结点t的错误代价Rt
    输入参数：tree_node:结点t； sample_size：整个数据集的大小
    输出参数：结点t的错误代价Rt
    '''
    counts = np.bincount(tree_node.data[:,-1])
    predict_class = np.argmax(counts)
    error_num = 0
    for i in range(tree_node.data.shape[0]):
        if tree_node.data[i,-1] != predict_class:
            error_num = error_num + 1
    Rt = error_num/sample_size
    return Rt

def predictLeavesErrorRTt(tree_node,sample_size):
    '''
    函数功能：计算子树Tt的错误代价RTt 
    输入参数：tree_node:结点t； sample_size：整个数据集的大小
    输出参数：子树Tt的错误代价RTt 
    '''
    RTt = 0
    if tree_node.left:
        RTt = RTt + predictLeavesErrorRTt(tree_node.left, sample_size)
    else:
        return predictErrorRt(tree_node,sample_size)
    if tree_node.right:
        RTt = RTt + predictLeavesErrorRTt(tree_node.right, sample_size)
    else:
        return 
    return RTt

def alphaCalculate(tree_node,sample_size):
    '''
    函数功能：计算子树的α值 
    输入参数：tree_node:结点t； sample_size：整个数据集的大小
    输出参数：子树的α值
    '''
    Rt = predictErrorRt(tree_node,sample_size)
    RTt = predictLeavesErrorRTt(tree_node,sample_size)
    alpha = (Rt - RTt)/(tree_node.leavesNum() - 1)
    return alpha

def alphaMin(Tree,sample_size):
    '''
    函数功能：遍历整棵树找到α的最小值 
    输入参数：tree:决策树的根结点； sample_size：整个数据集的大小
    输出参数：α的最小值
    '''
    if Tree.left:
        alpha_min = alphaCalculate(Tree, sample_size)
        alpha_left = alphaMin(Tree.left, sample_size)
        if alpha_min > alpha_left:
            alpha_min = alpha_left
        alpha_right = alphaMin(Tree.right, sample_size)
        if alpha_min > alpha_right:
            alpha_min = alpha_right
        return alpha_min
    else:
        return 100  #100超出alpha范围即忽略这一项  

def prunTree(Tree,sample_size,alpha_min):
    '''
    函数功能：将目前具有α最小值的子树剪枝 
    输入参数：tree:决策树的根结点； sample_size：整个数据集的大小； alpha_min：上个函数得到的α最小值
    输出参数：剪枝后的决策树
    '''
    prun_tree = BinNode(Tree.data,Tree.left,Tree.right,Tree.position)
    if prun_tree.left:
        alpha = alphaCalculate(prun_tree, sample_size)
        if alpha == alpha_min: #alpha有多个最小的情况未考虑
            prun_tree.setLeft(None)
            prun_tree.setRight(None)
            return prun_tree
        else:
            prun_tree.left = prun(prun_tree.left,sample_size,alpha_min)
            prun_tree.right = prun(prun_tree.right,sample_size,alpha_min)
    else:
        return prun_tree
    return prun_tree

def prun(prun_tree_list):
    '''
    函数功能：循环剪去具有α最小值的子树
    输入参数：开始时只有整棵决策树的列表
    输出参数：循环剪去最小α子树后的，一系列剪枝后的树，形成一个列表
    '''
    sample_size = prun_tree_list[0].data.shape[0]
    num = 0
    print('剪枝树T%d为：'%num)
    drawTree(prun_tree_list[-1])
    print()
    while prun_tree_list[-1].leavesNum() > 1:
        alpha_min = alphaMin(prun_tree_list[-1], sample_size)
        prun_tree_list.append(prunTree(prun_tree_list[-1], sample_size, alpha_min))
        num = num + 1
        print('剪枝树T%d为：'%num)
        drawTree(prun_tree_list[-1])
        print()
    return prun_tree_list

def decideTheBestTree(test_set,prun_tree_list):
    '''
    函数功能：找到最优决策树
    输入参数：test_set：测试集； prun_tree_list：剪枝树列表
    输出参数：对于这个测试集而言最优的决策树
    '''
    correct_rate_max = predictCorrectRate(test_set, prun_tree_list[0])
    best_tree = prun_tree_list[0]
    
    for i in range(len(prun_tree_list)):
        correct_rate = predictCorrectRate(test_set, prun_tree_list[i])
        if correct_rate > correct_rate_max:
            correct_rate_max = correct_rate
            best_tree = prun_tree_list[i]
    print('对于该测试集，最佳剪枝树是：')
    drawTree(best_tree)
    return best_tree

######
#下面的函数用来画树
def setTreeDepth(node,root_depth):
    if node.left:
        node.depth = root_depth
        root_depth = root_depth - 1
        setTreeDepth(node.left, root_depth)
        setTreeDepth(node.right, root_depth)
    else:
        node.depth = root_depth

def getDepthList(node): 
    depth_list = []
    node_list = []
    if node.left:
        depth_left_list,node_left_list = getDepthList(node.left)
        for i in range(len(depth_left_list)):
            depth_list.append(depth_left_list[i])
        for i in range(len(node_left_list)):
            node_list.append(node_left_list[i])
        depth_right_list,node_right_list = getDepthList(node.right)
        for i in range(len(depth_right_list)):
            depth_list.append(depth_right_list[i])
        for i in range(len(node_right_list)):
            node_list.append(node_right_list[i])
        depth_list.append(node.depth)
        node_list.append(node)
        return depth_list,node_list
    else:
        depth_list.append(node.depth)
        node_list.append(node)
        return depth_list,node_list
def drawTree(root):
    setTreeDepth(root,root.rootDepth())
    depth_list, node_list = getDepthList(root)
    depth_max = max(depth_list)
    for j in range(depth_max+1):
        for i in range(len(depth_list)):
            if depth_list[i] == depth_max:
                if node_list[i].position != None and node_list[i].left:
                    if node_list[i].position == 0:
                        word = '是否有房'
                    if node_list[i].position == 1:
                        word = '是否结婚'
                    print(' '*(3*pow(2,(depth_max+1))),\
                          word,end = '')
                else:
                    if node_list[i].predict() == 0:
                        predict_class = '没拖欠'
                    if node_list[i].predict() == 1:
                        predict_class = '拖欠' 
                    print(' '*(3*pow(2,(depth_max+1))),\
                          predict_class,end = '')
        if depth_max != 0:
            print()
            print(' '*(2*pow(2,(depth_max+1))+2),'/否',' '*(2*pow(2,(depth_max+1))),'\是')
            depth_max = depth_max - 1

if __name__ == '__main__':
    raw_dataset = readData('shuju1.csv')

    print('*'*50)
    Tree = generateTree(raw_dataset)
    drawTree(Tree)
    print()
    print('*'*50)
    print('下面进行预测')
    print('预测[0,0]分类为：',predictClass([0,0], Tree))
    print('预测[0,1]分类为：',predictClass([0,1], Tree))
    
    print('*'*50)
    print('下面开始使用代价复杂度（CCP）剪枝')
    prun_tree_list = []
    prun_tree_list.append(Tree)
    prun_tree_list = prun(prun_tree_list)
    test_set = readData('test1.csv')
    decideTheBestTree(test_set,prun_tree_list)
    print()
    print('即由于测试集特殊，设定为全部都是类0，所以剪枝后发现最好的剪枝树就是Tm，直接判断所有都没拖欠')

    
    
    
    
    
    
    
    
    
    
    

