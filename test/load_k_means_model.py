# 加载模型
import joblib
import numpy as np

loaded_model = joblib.load('kmeans_model.pkl')

# 预测新数据点的簇
new_data = np.array([[1, 2], [2.3, 4.5], [1.0, 3.5], [7.8, 9.0]])
predicted_labels = loaded_model.predict(new_data)

print(predicted_labels)