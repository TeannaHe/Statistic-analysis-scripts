from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np

def FDR_filter(df,meta_df,group_column_name,group1_name,group2_name):
    """
    FDR feature selection

    df: dataFrame,行为样本，列为feature
    meta_df: dataFrame, group information
    group_column_name: 分组列信息
    group1_name: 第一个组别的名称
    group2_name: 第二个组别的名称
    """
    subDf = df.loc[meta_df.index.tolist()]
    group1_sampleList = meta_df[meta_df[group_column_name] == 1].index.tolist()
    group2_sampleList = meta_df[meta_df[group_column_name] == 0].index.tolist()

    group1_df = df.loc[group1_sampleList]
    group2_df = df.loc[group2_sampleList]

    features = df.columns.tolist()
    
    pairs = list()
    for feature in features:
        x = group1_df[feature].values
        y = group2_df[feature].values
        pairs.append((x,y))
    
    #pvals = [ranksums(x, y).pvalue for x, y in pairs]
    pvals = [mannwhitneyu(x, y, alternative='two-sided').pvalue for x, y in pairs]
    fdr_results = multipletests(pvals, method='fdr_bh')
    fdr = fdr_results[1]  # FDR值
    rejected = fdr_results[0]  # 是否拒绝原假设（是否显著）
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'feature': features,
        'pvalue': pvals,
        'fdr': fdr,
        'significant': rejected,
        f'{group1_name}_mean': [np.mean(x) for x, y in pairs],
        f'{group2_name}_mean': [np.mean(y) for x, y in pairs],
        'mean_diff': [np.mean(x) - np.mean(y) for x, y in pairs]
        })
    
    return (results_df)
