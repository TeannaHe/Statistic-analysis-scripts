import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def pairWiseCompare(data1,data2):
    """
    data1:第一组数据，输入格式为dataframe
    data2:第二组数据，输入格式为dataframe
    输出为 mann-whitney U的u, p, data1的median以及data2的median
    """
    # NaN值用中位数填充
    data1 = data1.fillna(data1.median())
    data2 = data2.fillna(data2.median())
    # 确保是数值类型
    data1 = pd.to_numeric(data1, errors="coerce")
    data2 = pd.to_numeric(data2, errors="coerce")
    if data1.sum() == 0:
        print (f'data1 sum is 0, pass.')
        pass
    elif data2.sum() ==0:
        print (f'data2 sum is 0, pass.')
        pass
    else:
        u_stat, p_value = mannwhitneyu(data1,data2,alternative='two-sided')
        #u = round(u_stat,1)
        u = u_stat
        #p = round(p_value,5)
        p = p_value
        data1_summary = data1.agg(['median','mean', 'std'])
        data2_summary = data2.agg(['median','mean', 'std'])
        return (u,p,data1_summary['median'],data2_summary['median'])
