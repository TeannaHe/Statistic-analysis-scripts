import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import textwrap
from scipy.stats import mannwhitneyu

def p_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

#--- 自动换行函数（新增） ---
def wrap_title(text, width=40):
    """自动按指定字符宽度换行"""
    return "\n".join(textwrap.wrap(text, width))

def boxplot_two_groups(data1,data2,plot_save_path,box_plot_name,title=None,y_axis=None,group1_name=None,group2_name=None):
    """
    data1:第一组数据，输入格式为dataframe
    data2:第二组数据，输入格式为dataframe
    plot_save_path:图片保存路径
    box_plot_name:图片名称
    title:所做分析数据名称，将出现在图片的title中
    y_axis:输入的数据类型，将出现在图片的y轴名称中
    group1_name:第一组数据分组名称，将出现在图中
    group2_name:第二组数据分组名称，将出现在图中
    """
    plot_save_path = plot_save_path or '.'
    box_plot_name = box_plot_name or 'boxplot.png'
    title = title or 'Boxplot'
    y_axis = y_axis or ''
    group1_name = group1_name or 'group1'
    group2_name = group2_name or'group2'
    # NaN值用中位数填充
    data1 = data1.fillna(data1.median())
    data2 = data2.fillna(data2.median())
    # 确保是数值类型
    data1 = pd.to_numeric(data1.squeeze(), errors="coerce")
    data2 = pd.to_numeric(data2.squeeze(), errors="coerce")
    # 排除异常情况
    if data1.sum() == 0:
        print (f'data1 sum is 0, pass.')
        return
    elif data2.sum() ==0:
        print (f'data2 sum is 0, pass.')
        return
    u_stat, p_value = mannwhitneyu(data1,data2,alternative='two-sided')
    star = p_to_star(p_value)
    #准备绘图数据
    df = pd.DataFrame({
        'Value': pd.concat([data1, data2], ignore_index=True),
        'Group': [group1_name]*len(data1) + [group2_name]*len(data2)
        })
    #######------绘制boxplot---------------------------
    # 设置风格
    sns.set_theme(style="white",font_scale=1.1)
    # 绘制箱型图
    plt.figure(figsize=(4.5,4))
    ax = sns.boxplot(
            data=df,
            x='Group', 
            y='Value', 
            width=0.5,
            linewidth=1.5,
            showfliers=False,
            boxprops=dict(edgecolor='black'),
            medianprops=dict(color='black',linewidth=2)
    #        whiskerprops=dict(linewidth=1.5),
    #        capprops=dict(linewidth=1.5),
            )
    sns.stripplot(
            data=df,
            x='Group',
            y='Value',
            color='black',
            alpha=0.5,
            size=4,
            jitter=0.25
            )
    ##--- 添加标题，并自行换行title ---
    ax.set_title(title,fontsize=13,weight='bold')
    ax.set_xlabel("")
    ax.set_ylabel(y_axis,fontsize=12)
#    sns.despine(trim=True)
#    ax.grid(False)
    # ---------------- P-value bracket ----------------
    ymax = df['Value'].max()
    y_min, y_max = ax.get_ylim()
    h = (y_max - y_min) * 0.05
    y = ymax + h
    # 括号
    ax.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c='black')
    # 星号
    ax.text(
            0.5,
            y + h * 1.1,
            star,
            ha='center',
            va='bottom',
            fontsize=14,
            weight='bold'
            )
    # ---------------- clean ----------------
    sns.despine(trim=True)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f"{plot_save_path}/{box_plot_name}", dpi=300, bbox_inches='tight')
    plt.close()
