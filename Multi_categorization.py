import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

# 读取数据
ifname = r"archive\ratings.csv"
ifname2 = r"archive\movies.csv"
ifname3 = r"archive\tags.csv"
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
text_genres_features = data['genres']
text_tag_features = data['tag']
numeric_userId_features = data['userId']
numeric_movieId_features = data['movieId']
target = data['rating']

#将分数集合分类进入计算
target_bins = np.floor(target / 1).astype(int)
temp = np.zeros((len(target_bins), 6))
for i, bin_value in enumerate(target_bins):
    temp[i, bin_value] = 1
# 将分类结果转换为 PyTorch 的 Tensor 对象，并设置类型为 int 类型
target = torch.Tensor(temp).long()
#print(target)

# 将文本特征转换为词袋特征
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b|\|')
text_tag_features = text_tag_features.fillna('nan')
text_genres_features = vectorizer.fit_transform(text_genres_features).toarray()
text_tag_features = vectorizer.fit_transform(text_tag_features).toarray()

# 对数值型特征进行最小-最大缩放归一化
scaler = MinMaxScaler()
numeric_userId_features = scaler.fit_transform(numeric_userId_features.values.reshape(-1, 1))
numeric_movieId_features = scaler.fit_transform(numeric_movieId_features.values.reshape(-1, 1))
text_genres_features = scaler.fit_transform(text_genres_features)
text_tag_features = scaler.fit_transform(text_tag_features)

# 将特征转换为Tensor
numeric_userId_features = torch.Tensor(numeric_userId_features)
numeric_movieId_features = torch.Tensor(numeric_movieId_features)
text_genres_features = torch.Tensor(text_genres_features)
text_tag_features = torch.Tensor(text_tag_features)

# 合并特征
features = torch.cat((numeric_userId_features,numeric_movieId_features,text_genres_features,text_tag_features), dim=1)

# 划分训练集和测试集

train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, random_state=42
)


class MovieRatingDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return len(self.target)

# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[-1,: , :])
        return out

# 设置超参数
# 输入特征维度，根据具体数据集而定
input_size = features.shape[1]
# 隐藏状态维度
hidden_size = 500
# 输出维度，即评分分类
output_size = 6
num_layers = 6

# 创建模型实例
model = RNNModel(input_size, hidden_size, output_size,num_layers)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02,weight_decay=0.1)

# 加载数据集（假设已经准备好了数据集）
train_dataset = MovieRatingDataset(train_features, train_target)  # 训练集
test_dataset = MovieRatingDataset(test_features, test_target)  # 测试集

# 创建数据加载器
#batch_size=32表示每个批次的大小为32，即每次训练模型时会输入32个数据样本。shuffle=True表示每个epoch开始时都会将训练数据打乱顺序
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
num_epochs = 10
batch_size = 32
sequence_length = 1
train_losses = []
test_losses = []
plt.figure(figsize=(8,6))

input_size = features.shape[1]  # 根据实际情况调整 input_size 的值
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()

        # 前向传播和计算损失
        outputs = model(batch_data.unsqueeze(0))
        #print("outputs:", type(outputs), outputs.shape)
        loss = criterion(outputs, batch_labels.float())

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_data.size(0)

    # 计算训练集的平均损失
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)
    # 在测试集上评估模型
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data.unsqueeze(0))
            loss = criterion(outputs, batch_labels.float())
            test_loss += loss.item() * batch_data.size(0)

    # 计算测试集的平均损失
    test_loss /= len(test_dataset)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

plt.plot(train_losses,label='train loss')
plt.plot(test_losses,label='test loss')
plt.legend()
plt.title("train and test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
# 使用训练好的模型进行预测
model.eval()
thetrue = []
predictions = []
the_total = 0
set_interval = 0.25
set_num = 0
with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        outputs = model(batch_data.unsqueeze(0))
        list1 = outputs.squeeze().tolist()
        list2 = batch_labels.squeeze().tolist()
        max_cols_list1 = [row.index(max(row)) for row in list1]  # 获取 list1 每一行最大值的索引列数
        max_cols_list2 = [row.index(max(row)) for row in list2]  # 获取 list2 每一行最大值的索引列数
        for col1, col2 in zip(max_cols_list1, max_cols_list2):
            the_total += 1
            if col1 == col2:
                set_num += 1

        predictions.extend(max_cols_list1)
        thetrue.extend(max_cols_list2)

# 输出预测结果
print(predictions[:10])
print(thetrue[:10])
df = pd.DataFrame({'List1': predictions, 'List2': thetrue})
excel_file = 'archive/true.csv'
df.to_csv(excel_file, index=False)
print('Test Accuracy: ',set_num / the_total)
#print("Predictions:", predictions)
