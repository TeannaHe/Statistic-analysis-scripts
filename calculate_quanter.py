import pandas as pd
import numpy as np

def calculate_quanter(data1,data2,group1_name=None,group2_name=None):
    """
    data1：输入 series 或 array
    data2：输入 series 或 array
    """
    group1_name = group1_name or 'group1'
    group2_name = group2_name or 'group2'
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
    return ValueDf
