import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

#对特征值进行z-score转换,行为样本，列为特征值
def z_score(df):
    """
    df: 输入dataframe格式，行为样本，列为特征值，是对每一个特征值做标准化
    输出为标准化之后的 dataframe
    """
    standardData = StandardScaler().fit_transform(df)
    outputDf = pd.DataFrame(data = standardData,index=df.index,columns=df.columns)
    return (outputDf)
