import numpy as np

def sigmoid_transform(df):
    """
    将数据转换成0-1范围的数值
    df: 输入dataframe
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    df_sigmoid = df.applymap(sigmoid)
    return (df_sigmoid)
