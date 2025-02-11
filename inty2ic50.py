import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def calculate_ic50_4pl(concentrations, activities, name):
    """
    使用四参数逻辑回归（4PL）计算 IC50，并绘制浓度-活性曲线。

    参数：
    concentrations (array-like): 测试分子的浓度数组（μM）。
    activities (array-like): 对应的 FP 活性数据。
    name (str): 保存图片的文件名。

    返回：
    dict: 包含 IC50、EC50、Hill 斜率的字典。
    """

    # 定义 4PL 逻辑回归函数
    def four_param_logistic(x, A, B, IC50, H):
        return A + (B - A) / (1 + (x / IC50) ** H)

    # 估计初始参数 p0
    def auto_p0(concentrations, activities):
        A_init = min(activities)  # 最低活性（最大抑制）
        B_init = max(activities)  # 最高活性（未抑制）
        IC50_init = np.median(concentrations)  # 中位数作为 IC50 初值
        Hill_Slope_init = 1  # 斜率初值
        return [A_init, B_init, IC50_init, Hill_Slope_init]

    # 进行 4PL 拟合
    try:
        p0 = auto_p0(concentrations, activities)
        popt, _ = curve_fit(four_param_logistic, concentrations, activities, p0=p0, maxfev=10000)
        A_fit, B_fit, IC50_fit, Hill_fit = popt
    except RuntimeError:
        print("⚠️ 4PL 拟合失败，请检查数据质量！")
        return {"Error": "Curve fitting failed"}

    # 生成浓度范围
    concentration_range = np.logspace(np.log10(min(concentrations)), np.log10(max(concentrations)), 100)
    fitted_activities = four_param_logistic(concentration_range, *popt)

    # 绘制拟合曲线
    plt.figure(figsize=(8, 6))
    plt.plot(concentrations, activities, 'o', label='Observed Data', markersize=8)
    plt.plot(concentration_range, fitted_activities, '-', label='Fitted 4PL Curve', color='red')
    plt.xscale('log')
    plt.xlabel('Concentration (μM)')
    plt.ylabel('FP-Activity')
    plt.title(f'4PL Fit - {name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'C:\\Users\\syan\\Documents\\record\\phd\\2025\\pin1-few-shot\\PIN1Screen\\{name}.png')

    print(f"📌 IC50: {IC50_fit:.3f} μM")
    print(f"📌 Hill Slope: {Hill_fit:.3f}")

    return IC50_fit, Hill_fit


# 示例用法：
concentrations = np.array([0.1, 0.3, 1, 3, 10, 30, 100])  # 浓度数据
activities = np.array([90, 75, 50, 30, 20, 10, 5])  # 活性数据
calculate_ic50_4pl(concentrations, activities, "Example_Molecule")
