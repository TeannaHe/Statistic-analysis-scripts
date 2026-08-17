import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from itertools import combinations
import numpy as np
from tqdm import tqdm

def filter_based_on_correlation(df, threshold=0.95):
    """
    相关性特征筛选方法
    """
    print("计算相关系数矩阵...")
    corr_matrix = df.corr().abs()  # 使用绝对值相关系数
    # 创建上三角矩阵
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    print("筛选高度相关特征...")
    to_drop = []
    # 按列遍历，保留第一个出现的特征，删除后续相关特征
    for i, column in enumerate(tqdm(upper.columns)):
        if column in to_drop:
            continue
        # 找出与当前特征高度相关的其他特征
        correlated = upper[column][upper[column] > threshold].index.tolist()
        to_drop.extend(correlated)
    # 去重
    to_drop = list(set(to_drop))
    selected_features = [col for col in df.columns if col not in to_drop]
    print(f"\n结果统计:")                       
    print(f"原始特征数: {len(df.columns)}")
    print(f"保留特征数: {len(selected_features)}")
    print(f"删除特征数: {len(to_drop)}")
    print(f"保留比例: {len(selected_features)/len(df.columns):.2%}")
    return selected_features
