# ---- 📋 场景 1：静态检测（快照诊断）燃气泄漏预测模型 ----
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. 生成模拟数据（包括温度、压力、流量和泄漏标签）
#--------------------------------------------------

np.random.seed(42)  # 固定随机数种子，确保结果可重现

# 正常工况数据（无泄漏）--- 决策树不关心时间戳
normal_data = {
  "温度(℃)": np.random.normal(25, 2, 500),
  "压力(MPa)": np.random.normal(1.2, 0.1, 500),
  "流量(m³/s)": np.random.normal(0.8, 0.05, 500),
  "泄漏状态": 0  # 0表示无泄漏
}

# 异常工况数据（泄漏发生）--- 决策树不关心时间戳
leak_data = {
  "温度(℃)": np.random.normal(28, 3, 500),
  "压力(MPa)": np.random.normal(0.9, 0.2, 500),
  "流量(m³/s)": np.random.normal(1.5, 0.3, 500),
  "泄漏状态": 1  # 1表示泄漏
}

# - 算法会从所有样本里，针对每个特征，尝试多个“阈值”
#  （比如温度 < 28℃，压力 > 1.15MPa）作为分裂点。=> 决策树 => 随机森林
#-----------------------------------------------------------------------
# 算法尝试以下分裂：
#
# 用温度 < 28℃ 分裂：
#   左边（温度 < 28）：样本 25℃、27℃（均无泄漏）
#   右边（温度 ≥ 28）：样本 29℃、30℃（均有泄漏）
#   纯度高，分裂效果好。
#
# 用压力 < 1.15MPa 分裂：
#   左边（压力 < 1.15）：样本 1.0、1.1（均有泄漏）
#   右边（压力 ≥ 1.15）：样本 1.2、1.3（均无泄漏）
#   纯度也高。
#
# 用流量 < 0.7 分裂：
#   左边（流量 < 0.7）：样本 0.5、0.6（均有泄漏）
#   右边（流量 ≥ 0.7）：样本 0.7、0.8（均无泄漏）
#   纯度高。
#
# 算法会计算每种分裂的“纯度指标”，选择其中提升最大的，
# 比如“温度 < 28℃”作为第一个判断条件。
#----------------------------------------------
# 最终决策树大致像这样（简化版）
#
# 温度 < 28℃
#  ├─ 是：无泄漏
#  └─ 否：进一步判断压力
#       └─ 压力 ≥ 1.15
#          ├─ 是：无泄漏
#          └─ 否：泄漏

# 合并数据
df_normal = pd.DataFrame(normal_data)
df_leak = pd.DataFrame(leak_data)
df = pd.concat([df_normal, df_leak], ignore_index=True)

# 2. 数据预处理
#--------------------------------------------------

# 洗牌数据（打乱顺序）
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 分离特征和标签
X = df[["温度(℃)", "压力(MPa)", "流量(m³/s)"]]
y = df["泄漏状态"]

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 创建并训练模型
model = RandomForestClassifier(
  n_estimators=100,  # 树的数量
  max_depth=5,       # 树的最大深度
  random_state=42
)
model.fit(X_train, y_train)

print('-- 训练 OK.')

# 6. 模型评估
#--------------------------------------------------

# 训练集评估
train_preds = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_preds)

# 测试集评估
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)

# 打印评估结果
print("\n🔥 燃气泄漏检测模型评估报告 🔥")
print(f"训练集准确率: {train_accuracy:.2%}")
print(f"测试集准确率: {test_accuracy:.2%}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, test_preds))
print("\n分类报告:")
print(classification_report(y_test, test_preds))

# 7. 模拟实时监测预测
print("\n🔔 实时监测预测演示 🔔")
# 随机生成一个传感器读数样本
samples = np.array([
  [26.5, 1.15, 0.82],    # 正常样本1
  [25.8, 0.85, 1.45],    # 泄漏样本
  [27.2, 1.18, 0.78],    # 正常样本2
  [30.1, 0.92, 1.35]     # 泄漏样本
])

# 对样本进行相同的标准化处理
samples_scaled = scaler.transform(samples)

# 使用模型进行预测
predictions = model.predict(samples_scaled)
leak_probabilities = model.predict_proba(samples_scaled)[:, 1] * 100  # 泄漏概率百分比

# 打印预测结果
for i in range(len(samples)):
  status = "🚨 泄漏警报！" if predictions[i] == 1 else "✅ 状态正常"
  print(f"样本 {i+1}: 温度={samples[i,0]}℃ | 压力={samples[i,1]}MPa | 流量={samples[i,2]}m³/s")
  print(f"  预测结果: {status} | 泄漏概率: {leak_probabilities[i]:.1f}%")
  print("─" * 60)