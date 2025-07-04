# ---- 📋 场景 1：静态检测（快照诊断）燃气泄漏预测模型 - 决策树 ----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import tree

# 模拟生成数据集（实际应用中替换为真实数据）
def generate_sample_data(num_samples=10000):
  np.random.seed(42)
  
  data = {
      'P_in': np.random.normal(4.0, 0.5, num_samples),  # 进水压力 (MPa)
      'P_out': np.random.normal(3.8, 0.5, num_samples),  # 出水压力 (MPa)
      'T_in': np.random.normal(85, 5, num_samples),     # 进水温度 (°C)
      'T_out': np.random.normal(75, 5, num_samples),     # 出水温度 (°C)
      'Flow': np.random.normal(100, 20, num_samples),    # 水流量 (m³/h)
      'ambient_temp': np.random.uniform(-20, 5, num_samples),  # 环境温度 (°C)
      'hour': np.random.randint(0, 24, num_samples),     # 小时
      'day': np.random.randint(1, 31, num_samples),      # 日期
      'pipe_id': np.random.choice(['A1', 'B2', 'C3'], num_samples)  # 管道ID
  }
  
  # 计算差压特征
  data['DeltaP'] = data['P_in'] - data['P_out']
  
  # 创建目标变量：泄漏标志 (1表示泄漏)
  # 泄漏条件：差压异常 + 温度变化异常 + 流量波动
  leakage_conditions = (
      (data['DeltaP'] > 0.3) | 
      (data['T_in'] - data['T_out'] > 15) |
      (np.abs(data['Flow'] - 100) > 30)
  )
  
  data['leak_flag'] = np.where(leakage_conditions, 1, 0)
  
  return pd.DataFrame(data)

# 1. 加载数据
df = generate_sample_data()  # 替换为 pd.read_csv("your_data.csv") 

# 2. 特征工程
df['temp_diff'] = df['T_in'] - df['T_out']  # 计算温差
df['flow_deviation'] = np.abs(df['Flow'] - 100)  # 计算流量偏差

# 3. 特征选择
features = [
    'P_in', 
    'P_out',
    'T_in',
    'T_out',
    'Flow',
    'DeltaP',         # 关键压力差特征
    'ambient_temp',   # 环境温度
    'temp_diff',      # 新增温差特征
    'flow_deviation'  # 新增流量偏差特征
    # 注意：时间特征需要特殊处理（见下文说明）
]

X = df[features]
y = df['leak_flag']

# 4. 数据预处理
# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 6. 创建并训练决策树模型
model = DecisionTreeClassifier(
    max_depth=5,        # 限制树深度防止过拟合
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',  # 处理不平衡数据
    random_state=42
)

model.fit(X_train, y_train)

# 7. 模型评估
y_pred = model.predict(X_test)

print("模型准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 8. 可视化决策树
plt.figure(figsize=(20, 12))
tree.plot_tree(
    model, 
    feature_names=features, 
    class_names=['Normal', 'Leakage'],
    filled=True, 
    rounded=True,
    proportion=True,
    max_depth=3  # 只显示前3层避免图像过大
)
plt.title("供暖管道泄漏预测决策树")
plt.savefig('pipe_leakage_decision_tree.png', dpi=300)
plt.show()

# 9. 特征重要性分析
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n特征重要性:")
print(importance)