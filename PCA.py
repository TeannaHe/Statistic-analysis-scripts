import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib import rcParams

## 指定字体文件路径（替换为你的实际字体路径）
##font_path = '/usr/local/lib/python3.8/dist-packages/matplotlib/mpl-data/fonts/ttf/思源宋体CN-Medium.ttf'
##font_path = '/data120/home/users/jiaying/.local/lib/python3.8/site-packages/reportlab/fonts/思源宋体CN-Medium.ttf'
#
## 设置字体属性
#font_prop = fm.FontProperties(fname=font_path, size=12)
#
## 设置matplotlib全局参数
#plt.rcParams['font.family'] = font_prop.get_name()
#plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

rcParams['font.sans-serif'] = [
        'Noto Serif CJK SC',
        'Noto Serif CJK TC',
        'Noto Serif CJK JP',
        'Noto Serif CJK KR'
        ]

rcParams['axes.unicode_minus'] = False  # 必须设置

# 设置全局字体大小
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 36
rcParams['axes.labelsize'] = 30
rcParams['legend.fontsize'] = 20
rcParams['legend.title_fontsize'] = 24

def PCA_plot(df,y,group=None,title=None,outputPath=None,outputName=None):
    
    """
    df: 输入的dataframe
    y: 包含分组信息的dataframe
    group: y中包含分组信息的列名
    title: 出现在图中的title
    outputPath: 输出路径
    outputName: 图片名称

    """
    
    group = group or 'group'
    title = title or 'PCA'
    outputPath = outputPath or '.'
    outputName = outputName or 'pca.png'

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=df.index)
    # 合并分组信息
    df_pca['group'] = y[group]
    
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
    
    groups = df_pca['group'].unique()
    n_groups = len(groups)
    group_colors = generate_colormap(n_groups)
    color_map = dict(zip(groups, group_colors))
    
    # 创建 PCA 散点图
    plt.figure(figsize=(10, 7))
    
    # 绘制散点图
    scatter_handles = []
    for group_name, group_data in df_pca.groupby('group'):
        scatter = plt.scatter(
                group_data['PC1'],
                group_data['PC2'],
                color=color_map[group_name],
                label=group_name,
                s=150,
                alpha=0.9,
                edgecolor='w',
                linewidth=1
                )
        scatter_handles.append(scatter)
    
    # 添加标签和标题
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=30,labelpad=15)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=30,labelpad=15)
    plt.title(f"{title}", fontsize=36,pad=20)
    
    # 添加图例
    legend = plt.legend(
            handles=scatter_handles,
            title=group ,
            frameon=True,
            loc='best',
            ncol=1 if n_groups <= 15 else 2,
            fontsize=20
            )
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(
            f"{outputPath}/{outputName}",  # 文件名
            dpi=300,          # 分辨率（默认100）
            bbox_inches="tight",  # 避免截断标签
            facecolor='white'  # 背景色
            )
    plt.close()  # 关闭图形，避免内存泄漏
