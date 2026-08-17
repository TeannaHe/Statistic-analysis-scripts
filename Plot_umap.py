import umap.umap_ as umap
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

def plot_umap(df,y,group=None,title=None,outputPath=None,outputName=None):
    """
    df: 输入dataframe 行为样本，列为feature
    y: 包含分组信息的dataframe
    group: y中包含分组信息的列名
    title: 出现在图中的title
    outputPath: 输出路径
    outputName: 图片名称
    """
    group = group or 'group'
    title = title or 'UMAP'
    outputPath = outputPath or '.'
    outputName = outputName or 'umap.png'

    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(df)
    
    umapDf = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'], index=df.index)
    umapDf[group] = y.loc[umapDf.index, group]
    
    ############ 颜色编码 ###########
    plt.figure(figsize=(10, 7))

    # 1. 自动生成颜色映射
    def generate_colormap(n_groups):
        """生成视觉区分度高的颜色映射"""
        if n_groups <= 10:
            # 使用tab10色系(适用于≤10组)
            return plt.cm.tab10(np.linspace(0, 1, n_groups))
        elif n_groups <= 20:
            # 使用tab20色系(适用于≤20组)
            return plt.cm.tab20(np.linspace(0, 1, n_groups))
        else:
            # 超过20组使用seaborn的husl色系
            return sns.color_palette("husl", n_groups)

    groups = umapDf[group].unique()
    n_groups = len(groups)
    group_colors = generate_colormap(n_groups)
    color_map = dict(zip(groups, group_colors))

    for i in groups:
        sub = umapDf[umapDf[group] == i]
        plt.scatter(
            sub['UMAP1'], sub['UMAP2'],
            s=80, alpha=0.9,
            color=color_map[i],
            label=i
        )
    
    plt.xlabel("UMAP1", fontsize=16)
    plt.ylabel("UMAP2", fontsize=16)
    plt.title(title, fontsize=20)
    
    plt.legend(title=group, fontsize=12, title_fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig(f"{outputPath}/{outputName}", dpi=300)
    plt.close()
    
