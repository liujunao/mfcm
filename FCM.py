# FCM算法
import copy
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import decimal


# 用于初始化隶属度矩阵U
MAX = 10000.0
# 用于结束条件
Epsilon = 0.00000001


def import_data_format_iris(file):
    """ 
    格式化数据，前四列为data
    """
    data = []
    with open(str(file), 'r') as f:
        for line in f:
            current = line.strip().split(",")
            current_dummy = []
            for j in range(0, len(current)-1):
                current_dummy.append(float(current[j]))
            data.append(current_dummy)
    return data


def import_data_format_wine(file):
    """ 
    格式化数据，前四列为data
    """
    data = []
    with open(str(file), 'r') as f:
        for line in f:
            current = line.strip().split(",")
            current_dummy = []
            for j in range(0, len(current)):
                current_dummy.append(float(current[j]))
            data.append(current_dummy)
    return data


def initialise_U(data, cluster_number):
    """
    这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
    随机化一个隶属度矩阵
    """
    global MAX
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            # 用于生成一个指定范围内的整数
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def distance(point, center, cleck):
    """
    该函数计算2点之间的距离
    cleck: true --> 欧氏距离; false --> 曼哈顿距离
    """
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        if cleck:
			# Euclidean
            dummy += abs(point[i] - center[i]) ** 2
        else:
			# Manhattan
            dummy += abs(point[i] - center[i])

    result = 0.0
    if cleck:
        result = math.sqrt(dummy)
    else:
        result = dummy
    return result


def end_conditon(U, U_old):
    """
    结束条件: 当U矩阵随着连续迭代停止变化时，触发结束
    """
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def normalise_U(U):
    """
    在聚类结束时使U模糊化，每个样本的隶属度最大的为1，其余为0
    """
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U


def fuzzy(data, cluster_number, m, cleck):
    """
    这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    参数是：簇数(cluster_number)、隶属度的因子(m)、欧氏距离(True)与曼哈顿距离(False)的选择布尔值
    m的最佳取值范围为[1.5，2.5]
	"""
    # 迭代次数
    ilteration_num = 0
    # 初始化隶属度矩阵U
    U = initialise_U(data, cluster_number)
    # 循环更新U
    while(True):
        # 迭代次数
        ilteration_num += 1
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    # 分子
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    # 分母
                    dummy_sum_dum += (U[k][j] ** m)
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)
        # 创建一个距离向量, 用于计算U矩阵
        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j], cleck))
            distance_matrix.append(current)

        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    # 分母
                    dummy += (distance_matrix[i][j] /
                              distance_matrix[i][k]) ** (2/(m-1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            break
    print("迭代次数：" + str(ilteration_num))
    U = normalise_U(U)
    return U


def checker_iris(final_location):
    """
    和真实的聚类结果进行校验比对
    """
    right = 0.0
    for k in range(0, 3):
        checker = [0, 0, 0]
        for i in range(0, 50):
            for j in range(0, len(final_location[0])):
                if final_location[i + (50*k)][j] == 1:
                    checker[j] += 1
        right += max(checker)
    print("正确聚类的数量：" + str(right))
    answer = right / 150 * 100
    return "准确度：" + str(answer) + "%"


def checker_wine(final_location):
    """
    和真实的聚类结果进行校验比对
    """
    right = 0.0
    for k in range(0, 3):
        checker = [0, 0, 0]
        for i in range(0, 50):
            for j in range(0, len(final_location[0])):
                if final_location[i + (50*k)][j] == 1:
                    checker[j] += 1
        right += max(checker)
    print("正确聚类的数量：" + str(right))
    answer = right / 177 * 100
    return "准确度：" + str(answer) + "%"

if __name__ == '__main__':
    # 加载数据
    data = import_data_format_iris("wine.txt")
    # data = import_data_format_iris("iris.txt")
    # 用于计算运行时间
    start = time.time()
    # 调用模糊C均值函数: 类数为 3，隶属度为 2
    final_location = fuzzy(data, 3, 2, False)
    # 计算运行时间
    print("用时：{0}".format(time.time() - start))
    # 对结果进行校对
    print(checker_iris(final_location))
