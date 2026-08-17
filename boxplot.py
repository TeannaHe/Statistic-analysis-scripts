import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

import textwrap

#def add_table_auto(ax, table_df):
#    """
#    自动放置表格在右边空白区域，避免和箱线图重叠
#    """
#    # 缩小原来的 axes，给右侧留空间
#    pos = ax.get_position()
#    ax.set_position([pos.x0, pos.y0, pos.width*0.7, pos.height])  # 缩小到70%宽度
#    # 绘制表格到右侧空白
#    table = plt.table(
#            cellText=table_df.values,
#            rowLabels=table_df.index,
#            colLabels=table_df.columns,
#            cellLoc="center",
#            rowLoc="center",
#            bbox=[0.72, 0.5, 0.25, 0.4]  # x0, y0, width, height，调整合适
#            )
#    table.auto_set_font_size(False)
#    table.set_fontsize(8)
#    table.scale(1, 1)
#    return table

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
    #u = round(u_stat,1)
    u = u_stat
    #p = round(p_value,5)
    p = p_value
    #组合summary表格
    ValueDf = pd.DataFrame({
        group1_name:[
            data1.quantile(0.75),
            data1.median(),
            data1.quantile(0.25)
            ],
        group2_name:[
            data2.quantile(0.75),
            data2.median(),
            data2.quantile(0.25)
            ]
        },index=['3/4','1/2(median)','1/4']).round(3)
    #准备绘图数据
    df = pd.DataFrame({
        'Value': pd.concat([data1, data2], ignore_index=True),
        'Group': [group1_name]*len(data1) + [group2_name]*len(data2)
        })
    #######------绘制boxplot---------------------------
    # 设置风格
    sns.set(style="whitegrid")
    # 绘制箱型图
    plt.figure(figsize=(6,4))
    ax = sns.boxplot(
            data=df,
            x='Group', 
            y=f'Value', 
            hue='Group',
            palette="Set2",
            legend=False)
    ##--- 添加标题，并自行换行title ---
    wrapped_title = wrap_title(f"{title}   U={u}   P={p}", width=40)
    ax.set_title(wrapped_title)
    # 添加标签
    ax.set_xlabel("Group")
    ax.set_ylabel(y_axis)
    ax.grid(False)
    # 自动根据左右空间选择表格放置位置
    #add_table_auto(ax, ValueDf)
    # 保存为 PNG 文件
    plt.tight_layout()
    plt.savefig(f"{plot_save_path}/{box_plot_name}", dpi=300, bbox_inches='tight')
    plt.close()
