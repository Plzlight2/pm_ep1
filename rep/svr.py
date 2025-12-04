from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import math

# 读取特征
df = pd.read_csv("../pm25_features.csv")
X = df[["S0avg", "DoLPavg", "AoLPavg"]].values
y = df["PM25"].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建 pipeline（标准化 + SVR）
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

# 网格搜索
params = {
    "svr__C": [10, 100, 300],
    "svr__gamma": ["scale", 0.1, 0.01],
    "svr__epsilon": [0.01, 0.1]
}

grid = GridSearchCV(model, params, cv=5, scoring="r2")
grid.fit(X_train, y_train)

# 最终训练
best_model = grid.best_estimator_

# 预测
y_pred = best_model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}, R²: {r2:.3f}")

import matplotlib.pyplot as plt

# 散点图：真实值 vs 预测值
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.7, label="Prediction")

# 画 y=x 的参考线
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal line (y=x)")

plt.xlabel("True PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("SVR PM2.5 Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
