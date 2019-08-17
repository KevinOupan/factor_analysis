"""通过因子分析，从泸深300中挑选18具有代表性的股票"""

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import math
from math import e


# 读取收盘价数据
datafile = u'E:\\桌面\\all_data\\my_data2.csv'
data = pd.read_csv(datafile)


def Return(data_mat):              # 通过收盘价计算收益率
    m = np.shape(data_mat)[0]
    n = np.shape(data_mat)[1]
    data_mat2 = pd.DataFrame(index=range(m-1), columns=range(n))
    for j in range(n):
        for i in range(m-1):
            data_mat2.iloc[i, j] = math.log(data_mat.iloc[i+1, j]/data_mat.iloc[i, j], e)
    return data_mat2


def Return1(data_mat):     # 去掉预期收益为负的股票
    m = np.shape(data_mat)[0]
    n = np.shape(data_mat)[1]
    return_exp = pd.Series(index=range(n))
    for j in range(n):
        return_exp[j] = np.mean(data_mat.iloc[:, j])
    ss = np.sum(return_exp > np.zeros(n))
    # print(ss)
    data_mat1 = pd.DataFrame(index=range(m), columns=range(int(ss)))
    j1 = 0
    for j2 in range(n):
        if np.mean(data_mat.iloc[:, j2]) > 0:
            data_mat1.iloc[:, j1] = data_mat.iloc[:, j2]
            j1 += 1
    return data_mat1


def normal(data_mat):                   # 对各只股票极差标准化
    m = np.shape(data_mat)[0]
    n = np.shape(data_mat)[1]
    data_mat2 = pd.DataFrame(index=range(m), columns=range(n))
    for j in range(n):
        max_col = max(data_mat.iloc[:, j])
        min_col = min(data_mat.iloc[:, j])
        for i in range(m):
            data_mat2.iloc[i, j] = (data_mat.iloc[i, j] - min_col)/(max_col - min_col)
    return data_mat2


def Coef_var(data_mat):                # 求各只股票的变异系数
    m = np.shape(data_mat)[0]
    n = np.shape(data_mat)[1]
    coef_col = pd.Series(index=range(n))
    for j in range(n):
        mean_col = np.mean(data_mat.iloc[:, j])
        var1 = 0
        for i in range(m):
            var1 += (data_mat.iloc[i, j] - mean_col)**2
        var_col = (var1/(m-1))**(1/2)
        coef_col[j] = var_col/mean_col*1.0
    return coef_col


# 通过变异系数求得各股票权重，并赋给标准化后的股票数据
data2 = Return(data)
data3 = Return1(data2)
data_normal = normal(data2)
coef_var = Coef_var(data_normal)
m = np.shape(data_normal)[0]
n = np.shape(data_normal)[1]
data_new = pd.DataFrame(index=range(m), columns=range(n))
for j in range(n):
    for i in range(m):
        data_new.iloc[i, j] = (coef_var[j]/sum(coef_var))*data_normal.iloc[i, j]

# data_new即是处理后的股票数据
# print(data_new)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# 建立模型
fa = FactorAnalyzer(rotation='varimax', n_factors=12)  # 固定公共因子个数为5
fa.fit(data_new)
print("公因子方差：\n", fa.get_communalities())  # 公因子方差
matrix_orth = fa.loadings_
print("\n成分矩阵\n", matrix_orth)
var = fa.get_factor_variance()  # 给出贡献率
print("\n解释的总方差(即贡献率):\n", var)
# 分别取两位小数
print("\n特征值：\n", list(map(lambda x: round(x, 4), var[0])))
print("\n因子贡献率：\n", list(map(lambda x: round(x, 4), var[1])))
print("\n累计贡献率：\n", list(map(lambda x: round(x, 4), var[2])))

# 设置数据框的最大行、最大列和不换行（针对数据框）
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('expand_frame_repr', False)
# 将数据类型转换为数据框
data22 = pd.DataFrame(data)
# 取出数据框的列名
columns_name = data22.columns
# 按因子分析找出相应的股票
for i in range(94):
    if (matrix_orth.iloc[i, 6] == sorted(matrix_orth.iloc[:, 6], reverse=True)[0]):
        print(i)
# 输出选择的股票
for i in [20, 46, 35, 81, 2, 90, 67, 72, 17, 48, 52, 8, 71, 61, 19, 1, 59, 6]:
    print(data22.columns[i])

