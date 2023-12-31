import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

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
text_features = data['genres'] + ' ' + data['tag']
numeric_features = data[['userId', 'movieId']]
target = data['rating']
print(data['rating'].mean())
# 将最终评分转换为二分类标签
target = torch.Tensor(target.values).float()
target = torch.where(target >= 3.5, 1, 0).unsqueeze(1)

# 将文本特征转换为词袋特征
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b|\|')
text_features = text_features.fillna('')
text_features = vectorizer.fit_transform(text_features).toarray()
print(text_features.shape)
# 对数值型特征进行最小-最大缩放归一化
scaler = MinMaxScaler()
numeric_features = scaler.fit_transform(numeric_features)
text_features = scaler.fit_transform(text_features)

# 将特征转换为Tensor
#text_features = torch.Tensor(text_features)
#numeric_features = torch.Tensor(numeric_features.values)
#numeric_features = torch.Tensor(numeric_features.values)
text_features = torch.Tensor(text_features)
numeric_features = torch.Tensor(numeric_features)

# 合并特征
features = torch.cat((text_features, numeric_features), dim=1)
# 划分训练集和测试集
train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, random_state=42
)