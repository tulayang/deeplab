# -- 预测锅炉出口温度
#
# 已知 [负载，环境温度，流量]，预测 (锅炉温度)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 500

# 负载：50~100
load = np.random.uniform(50, 100, n)  

# 环境温度：-10~35
env_temp = np.random.uniform(-10, 35, n)     

# 流量：80~180
steam_flow = np.random.uniform(80, 180, n)         

# 温度 = 0.6*load - 0.3*env_temp + 0.2*steam_flow + 噪声
outlet_temp = (
    0.6 * load -
    0.3 * env_temp +
    0.2 * steam_flow +
    np.random.normal(0, 3, n)
)

# 构造模拟数据（3 个特征 - [负载，环境温度，流量]）
#-------------------------------------------------------
df = pd.DataFrame({
    'load': load,              # 负载：50~100
    'env_temp': env_temp,      # 环境温度：-10~35
    'steam_flow': steam_flow,  # 流量：80~180
    'outlet_temp': outlet_temp # 温度 = 0.6*load - 0.3*env_temp + 0.2*steam_flow + 噪声
})

X = df[['load', 'env_temp', 'steam_flow']] # 特征
y = df['outlet_temp']                      # 标签

# 拆分为训练集（80%）、测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 训练模型
#-------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# 查看模型结果
#-------------------------------------------------------
# 每个输入特征的系数（coef）
# 系数是模型对每个特征的“加权影响力”，假如我们得到这个模型：
#   outlet_temp = 0.59*load - 0.31*env_temp + 0.20*steam_flow + 1.72
# 可以理解为：
#   每增加1单位的“负载”，温度 ↑ 0.59 度
#   每升高1度“环境温度”，温度 ↓ 0.31 度
#   每增加1单位“流量”，  温度 ↑ 0.20 度
# 这就是模型“怎么想”的过程！每个特征都被它“打上了分值”。
coef = model.coef_            

# 一个截距项（intercept）    
# 模型在所有输入为 0 时的预测值
# 在实际中它没那么重要，更多是一个“基础值”或者“偏移量”                     
intercept = model.intercept_  

# 输出：
#       load  的系数是：0.61
#   env_temp  的系数是：-0.29
# steam_flow  的系数是：0.20    
#
# 也就是 outlet_temp = 0.61*load - 0.29*env_temp + 0.20*steam_flow - 1.11
for name, c in zip(X.columns, coef):
  print(f"{name:>12s}  的系数是：{c:.2f}")

# 输出：-1.11
print(f"\n截距项（bias）：{intercept:.2f}")

# 模型预测与评估
#-------------------------------------------------------
from sklearn.metrics import mean_squared_error, r2_score

# 预测测试集的所有数据
#-------------------------------------------------------
y_pred = model.predict(X_test)

# mean_squared_error（简称 MSE）衡量回归模型预测精度的常用指标之一
# 告诉我们：模型预测值与真实值之间的差距有多大，越小越好。
mse = mean_squared_error(y_test, y_pred)

# R² （决定系数） 是一个衡量回归模型预测准确度的指标
# 它告诉我们模型能解释数据中多少比例的变异性。
# 它衡量了模型对数据的拟合程度，值越接近 1，说明模型表现越好。
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R² 分数: {r2:.2f}")

# 预测新数据
#-------------------------------------------------------
new_data = np.array([[55,-3,174], [65,-1,188]])
predicted_temp = model.predict(new_data)
print(f"预测的锅炉出口温度是：{predicted_temp[0]:.2f} °C")
print(f"预测的锅炉出口温度是：{predicted_temp[1]:.2f} °C")