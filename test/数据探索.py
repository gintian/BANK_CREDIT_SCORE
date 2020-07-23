import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# 忽略警告
warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_csv(r"K:\DataModel\CREDIT_SCORE\data\df_training.csv", encoding='utf-8')

# 划分数据
df0 = data.iloc[:, 0]
df1 = data.iloc[:, 1:-1]
df2 = data.iloc[:, -1]

# 获取列名
columns = data.columns.tolist()[1: -1]

# 缺失值处理
imputer = SimpleImputer(np.nan, 'mean')
df3 = pd.DataFrame(imputer.fit_transform(df1), columns=columns)

# 合并数据集
data = pd.concat([df3, df2], axis=1)


# 异常值处理
# def Outliers(data, x):
#     Percentile = np.percentile(data[x], [0, 25, 50, 75, 100])
#     IQR = Percentile[3] - Percentile[1]
#     UpLimit = Percentile[3] + IQR * 1.5
#     DownLimit = Percentile[1] - IQR * 1.5
#     data[x] = data[x][data[x].between(DownLimit, UpLimit)]
#     return data[x]
#
# # 剔除异常值，使用四分位上下限
# for i in columns:
#     data[i] = Outliers(data, i)


# 输出各字段分布情况图
# 大多数字段明显偏态，后续建模需考虑纠偏处理
# data.plot(kind='box', subplots=True, layout=(6,6), sharex=False)
#
# plt.show()

valia = data.columns.tolist()

va = "+%s" * 27
fomula = "TARGET~%s"+ va
fomula = fomula % tuple(valia[:-1])

lm = smf.ols(fomula, data).fit()
print(lm.summary())




