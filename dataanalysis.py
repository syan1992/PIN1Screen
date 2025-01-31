import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from inty2ic50 import calculate_hill_slope_ic50

def draw_boxplot(data, x_label, y_label):
    """
    绘制箱线图
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x_label, y=y_label, data=data, palette="Set2")

    # 添加标题和标签
    plt.title(f"Boxplot of {y_label} for {x_label}=0 and {x_label}=1", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # 显示图表
    plt.show()

def data_split(data, features, target):
    """
    按照 features 进行数据划分，并 stratify 目标变量 y，同时保留所有列
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, data[target], test_size=0.2, stratify=data['y'], random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )

    return X_train, X_test, X_val, y_train, y_test, y_val

# 读取数据
input_data = pd.read_csv('..\\dataset\\NIH_FP.csv')

input_data['y'] = input_data['PUBCHEM_ACTIVITY_OUTCOME'].apply(lambda x: 1 if x == 'Active' else 0)
input_data['P_E'] = (input_data['Potency'] / input_data['Efficacy']) ** 0.5

ic_columns = [
    'FP-Activity at 0.00368 uM',
    'FP-Activity at 0.459 uM',
    'FP-Activity at 2.296 uM',
    'FP-Activity at 11.52 uM',
    'FP-Activity at 57.34 uM',
    'FP-Activity at 114.8 uM'
    ]

ic50 = []
hill_slope = []
ec50 = []
ac50 = []

concentrations = np.array([0.00368, 0.459, 2.296, 11.52, 57.34, 114.8])
for i in range(len(input_data)):
    activity = input_data[ic_columns].iloc[i].values
    dropnan = np.where(~np.isnan(activity))[0]
    hs_i, ec_i, ic_i, ac_i = calculate_hill_slope_ic50(concentrations[dropnan], activity[dropnan], i)
    ic50.append(ic_i)
    hill_slope.append(hs_i)
    ec50.append(ec_i)
    ac50.append(ac_i)

input_data["IC50"]= ic50
input_data["Hill_Slope"] = hill_slope
input_data["EC50"] = ec50
input_data["AC50"]= ac50

input_data.to_csv('..\\dataset\\NIH_FP_IC50.csv', index=False)

'''
# 选择要进行对数变换的列
activity_columns = [
    'Total_fluor_inty-Activity at 0.459 uM',
    'Total_fluor_inty-Activity at 2.296 uM',
    'Total_fluor_inty-Activity at 11.52 uM',
    'Total_fluor_inty-Activity at 57.34 uM',
    'Total_fluor_inty-Activity at 114.8 uM',
    'Potency',
    'Efficacy'
]
'''

X_train, X_test, X_val, y_train, y_test, y_val = data_split(input_data, features=ic_columns, target='y')

# **保存完整的数据集，包括所有列**
X_train.to_csv('pin1_fp\\raw\\train_pin1_fp_1.csv', index=False)
X_test.to_csv('pin1_fp\\raw\\test_pin1_fp_1.csv', index=False)
X_val.to_csv('pin1_fp\\raw\\valid_pin1_fp_1.csv', index=False)

print("数据集已成功划分并保存：")
print("Train set:", X_train.shape)
print("Test set:", X_test.shape)
print("Validation set:", X_val.shape)
