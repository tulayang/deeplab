from sklearn.linear_model import LinearRegression
import numpy as np

# 负载（x）- [压力]
x = np.array([[60], [70], [80], [90], [100]])

# 出口温度（y）
y = np.array([120, 130, 140, 150, 160])

# 训练魔性 - 线性回归
model = LinearRegression()
model.fit(x, y)

# 预测趋势
predict_temp = model.predict([[85]])
print(f"预测负载[85]时的温度是：{predict_temp[0]:.2f} °C")