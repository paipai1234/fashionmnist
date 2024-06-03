import torch
import torchvision
from torchvision import datasets, transforms
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

transform = transforms.Compose([transforms.ToTensor()])
# 1. 加载FashionMNIST数据集
train_dataset = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transform)
print("数据加载完成")

def dataset_to_numpy(dataset):
    data = []
    targets = []
    for img, label in dataset:
        data.append(np.array(img).squeeze())  # 去除单通道维度
        targets.append(label)
    return np.array(data), np.array(targets)


train_data, train_labels = dataset_to_numpy(train_dataset)
test_data, test_labels = dataset_to_numpy(test_dataset)

# 2. 提取HOG特征
def extract_hog_features(data):
    hog_features = []
    for img in data:
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(fd)
    return np.array(hog_features)


print("提取hog特征")
train_hog_features = extract_hog_features(train_data)
test_hog_features = extract_hog_features(test_data)

# 标准化HOG特征
scaler = StandardScaler()
train_hog_features = scaler.fit_transform(train_hog_features)
test_hog_features = scaler.transform(test_hog_features)

# 3. 训练SVM
print("训练svm")
clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
clf.fit(train_hog_features, train_labels)


# 4. 测试模型
test_predictions = clf.predict(test_hog_features)

print("模型测试中")
# 打印分类报告和准确率
print("Classification report for SVM classifier:")
print(classification_report(test_labels, test_predictions))
print("Accuracy:", accuracy_score(test_labels, test_predictions))
# 计算混淆矩阵
conf_matrix = confusion_matrix(test_labels, test_predictions)

# 打印混淆矩阵
print("Confusion Matrix:")
print(conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()