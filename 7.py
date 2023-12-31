import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import torch.nn.functional as F


# 读取数据
ifname = r"archive\ratings.csv"
ifname2 = r"archive\movies.csv"
ifname3 = r"archive\tags.csv"
df = pd.read_csv(r"D:\QQ\Download\output1.csv", encoding='gbk')
movie_data = pd.read_csv(ifname2)
tags_data = pd.read_csv(ifname3)
data = pd.read_csv(ifname)
data = data.drop('timestamp', axis=1)
merged_data = pd.merge(data, tags_data, on=['userId', 'movieId'], how='left')
merged_data_second = pd.merge(merged_data, movie_data, on='movieId', how='left')
merged_data_second.drop('title', axis=1, inplace=True)
merged_data_second.drop('timestamp', axis=1, inplace=True)
data = merged_data_second
# 特征选择
data['tag'].fillna('', inplace=True)
text_features = data['genres'] + ' ' + data['tag']
numeric_features = data[['userId', 'movieId']]
target = data['rating']
print(data['rating'].mean())
# 将最终评分转换为二分类标签
target = torch.Tensor(target.values).float()
target = torch.where(target >= 3.5, 1, 0).unsqueeze(1)

# 将文本特征转换为词袋特征
vectorizer = CountVectorizer()
text_features = text_features.fillna('')
text_features = vectorizer.fit_transform(text_features).toarray()

# 将特征转换为Tensor
text_features = torch.Tensor(text_features)
numeric_features = torch.Tensor(numeric_features.values)

# 合并特征
features = torch.cat((text_features, numeric_features), dim=1)
print(features)
# 划分训练集和测试集
train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 定义模型
class ClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        #防止过拟合每次丢失0.2神经元
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = ClassificationModel(features.shape[1])

# 定义损失函数和优化器
criterion = nn.BCELoss()
#优化器加上学习率以及惩罚率  两者的参数可以调整寻找到最佳
optimizer = optim.Adam(model.parameters(), lr=0.01,weight_decay=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_target.float())
    loss.backward()
    optimizer.step()

    # 每10个epoch打印一次损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 在测试集上进行预测
model.eval()  # 设置模型为评估模式，即预测模式

with torch.no_grad():
    test_outputs = model(test_features)
    test_pred = torch.where(test_outputs >= 0.5, 1, 0)
    accuracy = accuracy_score(test_target, test_pred)
    precision = precision_score(test_target, test_pred)
    recall = recall_score(test_target, test_pred)
    f1 = f1_score(test_target, test_pred)
    print(f'Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')