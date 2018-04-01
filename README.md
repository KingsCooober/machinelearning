# machinelearning

## start machinelearning

## add KNN Algorithm for machinelearning project.

## add model_selection for machinelearning project

# Note

## Data Normalization

- 数据标准化是将是将数据按比例缩放，使之落入一个小的特定区间。在某些比较和评价的指标处理中经常会用到，去除数据的单位限制，将其转化为无量纲的纯数值，便于不同单位或量级的指标能够进行比较和加权。其中最典型的就是数据归一化处理，即将数据统一映射到[0,1]区间上。

### 归一化的目标

1. 把数变为（0, 1）之间的小数
    主要是为了数据处理方便提出来的，把数据映射到0～1范围之内处理，更加便捷快速，应该归到数字信号处理范畴之内。
2. 把有量纲表达式变为无量纲表达式
    归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量。
## Max-Min Normalization

- 也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1]之间。转换函数如下：
    x* = (x - min)/(max - min)

- 其中max为样本数据的最大值，min为样本数据的最小值。这种方法有个缺陷就是当有新数据加入时，可能导致max和min的变化，需要重新定义。

## Zero-Mean Normalization

- 这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。经过处理的数据符合标准正态分布，即均值为0，标准差为1，转化函数为：
    x* = (x - u)/d
- 其中u为所有样本数据的均值，d为所有样本数据的标准差。

