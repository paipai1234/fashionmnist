import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt

# 数据转换器
transform = transforms.Compose([transforms.ToTensor()])

# 加载FashionMNIST数据集
train_dataset = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transform)

print("数据加载完成")

# 将数据集转换为numpy数组
def dataset_to_numpy(dataset):
    data = []
    targets = []
    for img, label in dataset:
        data.append(np.array(img).squeeze())  # 去除单通道维度
        targets.append(label)
    return np.array(data), np.array(targets)

train_data, train_labels = dataset_to_numpy(train_dataset)
test_data, test_labels = dataset_to_numpy(test_dataset)

# 将图像展平为一维数组
n_samples = len(train_dataset)
train_data = train_data.reshape((n_samples, -1))
n_samples_test = len(test_dataset)
test_data = test_data.reshape((n_samples_test, -1))

# 标准化像素特征
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


print("开始训练SVM")
clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
clf.fit(train_data, train_labels)

# 使用最佳参数在测试集上进行评估
print("模型测试中")
predicted = clf.predict(test_data)

print("分类报告：")
print(metrics.classification_report(test_labels, predicted))

# 计算混淆矩阵
conf_matrix = metrics.confusion_matrix(test_labels, predicted)

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
